"""Train the GNN recommender per ADR 008, logging to MLflow.

    uv run python -m recsys.training.train_gnn --setup validation --seed 0
    uv run python -m recsys.training.train_gnn --setup test --seed 0 --epochs 3   # fixed epochs, no eval

Validation setup: training graph 9-11 Nov, examples 12-13 Nov, early stopping on the
14 Nov test-mix weighted AUC scored with the 9-13 Nov graph (patience from config).
Test setup: training graph 9-12 Nov, examples 13-14 Nov, a fixed number of epochs (the best
validation epoch count); scoring 15 Nov with the 9-14 Nov graph happens in Part G.

Loss: softmax cross-entropy over 1 clicked + K non-clicked articles from the same impression,
with a learnable temperature (logit scale). The scale multiplies every score by the same
positive number, so it changes the loss but never a ranking.
"""

import argparse
import os
import dataclasses
import subprocess
import time

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from recsys.data.mind import explode_impressions
from recsys.data.splits import known_user_share
from recsys.evaluation import evaluate
from recsys.evaluation.harness import segment_metrics
from recsys.graph.periods import all_behaviors, load_period, period_split
from recsys.models.gnn import GNNEncoder, UserEncoder, encode_all_articles, load_model_config, score
from recsys.models.history import history_tensor, impression_histories
from recsys.paths import PROCESSED_DIR, PROJECT_ROOT

MODELS_DIR = PROCESSED_DIR / "models"
EVAL_SAMPLING_SEED = 1234  # inference also samples neighbours; fix it so evaluations are repeatable


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


class ClickExamples:
    """One example per click in the example period: (impression, clicked article, negative pool)."""

    def __init__(self, examples: pd.DataFrame, article_row: pd.Series, histories: torch.Tensor):
        rows, imp_pos = [], pd.Series(np.arange(len(examples)), index=examples["impression_id"])
        for imp, cands, labels in examples[["impression_id", "candidates", "labels"]].itertuples(index=False):
            cand_rows = article_row.loc[cands].to_numpy()
            pos = cand_rows[np.asarray(labels) == 1]
            neg = cand_rows[np.asarray(labels) == 0]
            rows += [(imp_pos[imp], p, neg) for p in pos]
        self.imp = torch.tensor([r[0] for r in rows])
        self.pos = torch.tensor([r[1] for r in rows])
        self.neg_pools = [r[2] for r in rows]
        self.histories = histories  # [n_impressions, max_history], aligned with `examples`

    def __len__(self) -> int:
        return len(self.pos)

    def negatives(self, idx: np.ndarray, k: int, rng: np.random.Generator) -> torch.Tensor:
        """k negatives per example; with replacement only if the impression has fewer than k."""
        return torch.tensor(np.stack([rng.choice(self.neg_pools[i], k, replace=len(self.neg_pools[i]) < k)
                                      for i in idx]))


class Recommender(nn.Module):
    def __init__(self, data, cfg: dict):
        super().__init__()
        self.encoder = GNNEncoder.from_data(data, cfg)
        self.users = UserEncoder(cfg["gnn"]["hidden_dim"])
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))  # learnable temperature


def _sampler(data, fanout) -> NeighborLoader:
    # Seeds are chosen per step; input_nodes = all articles so that index == article row.
    return NeighborLoader(data, num_neighbors={et: list(fanout) for et in data.edge_types},
                          input_nodes=("article", torch.arange(data["article"].num_nodes)), batch_size=1)


def train_step(model, loader, ex: ClickExamples, idx: np.ndarray, k: int, rng) -> torch.Tensor:
    cand = torch.cat([ex.pos[idx].unsqueeze(1), ex.negatives(idx, k, rng)], dim=1)  # [B, 1 + k]
    hist = ex.histories[ex.imp[idx]]                                                # [B, L]
    seeds = torch.unique(torch.cat([cand.flatten(), hist[hist >= 0]]))             # sorted article rows

    batch = loader.collate_fn(seeds)
    if not isinstance(batch, HeteroData):  # older PyG: collate returns raw sampler output
        batch = loader.filter_fn(batch)
    assert torch.equal(batch["article"].n_id[: len(seeds)], seeds)
    vec = model.encoder(batch)["article"][: len(seeds)]

    local = lambda rows: torch.searchsorted(seeds, rows)
    hist_local = torch.where(hist >= 0, local(hist.clamp_min(0)), torch.full_like(hist, -1))
    users = model.users(vec, hist_local)                                            # [B, d]
    logits = model.log_scale.exp() * (users.unsqueeze(1) * vec[local(cand)]).sum(-1)
    return F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long))         # the click is column 0


class Scorer:
    """Scores a target split with a given scoring graph (histories computed once)."""

    def __init__(self, eval_split, graph, ids, max_history: int, fanout):
        self.split, self.graph, self.fanout = eval_split, graph, fanout
        self.row = pd.Series(np.arange(len(ids["article"])), index=ids["article"])
        h = impression_histories(eval_split.target, eval_split.train, max_history)
        self.hist = history_tensor(h, self.row, max_history)
        self.user_pos = pd.Series(np.arange(len(h)), index=h.index)
        self.pairs = explode_impressions(eval_split.target)[["impression_id", "news_id"]]

    @torch.no_grad()
    def scores(self, model) -> pd.DataFrame:
        model.eval()
        with torch.random.fork_rng():
            torch.manual_seed(EVAL_SAMPLING_SEED)
            vec = encode_all_articles(model.encoder, self.graph, fanout=self.fanout)
        users = model.users(vec, self.hist)
        s = score(users[self.user_pos.loc[self.pairs["impression_id"]].to_numpy()],
                  vec[self.row.loc[self.pairs["news_id"]].to_numpy()])
        return self.pairs.assign(score=s.numpy())


def validation_eval_split():
    """14 Nov scored against the 9-13 Nov period, with the test split's known-user share."""
    behaviors = all_behaviors()
    test = period_split("test_score", behaviors)
    return dataclasses.replace(period_split("validation_score", behaviors), name="validation",
                               reweight_known_share=known_user_share(test.train, test.target))


def train(setup: str, seed: int, epochs: int | None = None, cfg: dict | None = None, run_name: str | None = None):
    """Train one model. With epochs=None: early stopping on validation (validation setup only)."""
    cfg = cfg or load_model_config()
    tc, fanout, max_hist = cfg["train"], cfg["gnn"]["fanout"], cfg["user"]["max_history"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    behaviors = all_behaviors()
    data, ids = load_period(f"{setup}_train")
    ex_split = period_split(f"{setup}_train", behaviors)  # train = graph period, target = example period
    row = pd.Series(np.arange(len(ids["article"])), index=ids["article"])
    clicks_source = pd.concat([ex_split.train, ex_split.target])  # clicks before t, from 9 Nov on
    histories = history_tensor(impression_histories(ex_split.target, clicks_source, max_hist), row, max_hist)
    examples = ClickExamples(ex_split.target, row, histories)

    model = Recommender(data, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    loader = _sampler(data, fanout)

    early_stop = epochs is None
    if early_stop:
        assert setup == "validation", "early stopping needs the validation day"
        eval_split = validation_eval_split()
        score_graph, score_ids = load_period("validation_score")
        scorer = Scorer(eval_split, score_graph, score_ids, max_hist, fanout)

    # Local file store (mlruns/). MLflow 3 marks it "maintenance mode" and asks for a database
    # backend, which mlflow-skinny can't provide; the documented opt-out keeps it working.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("gnn")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=run_name or f"{setup}-seed{seed}") as run:
        mlflow.log_params({"setup": setup, "seed": seed, "git_commit": git_commit(), "examples": len(examples),
                           "fanout": str(fanout), "max_history": max_hist,
                           **{f"gnn.{k}": v for k, v in cfg["gnn"].items()},
                           **{f"train.{k}": v for k, v in tc.items()}})
        best_auc, best_epoch, best_state, bad = -1.0, 0, None, 0
        for epoch in range(1, (epochs or tc["max_epochs"]) + 1):
            model.train()
            order, losses, t0 = rng.permutation(len(examples)), [], time.time()
            for start in range(0, len(order), tc["batch_size"]):
                loss = train_step(model, loader, examples, order[start: start + tc["batch_size"]], tc["negatives"], rng)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            metrics = {"train_loss": float(np.mean(losses)), "epoch_seconds": time.time() - t0,
                       "logit_scale": model.log_scale.exp().item()}

            if early_stop:
                res = evaluate(scorer.scores(model), eval_split, model="gnn").set_index("segment")
                # MLflow metric names can't contain "=": "user=known" -> "user_known"
                metrics |= {f"val_auc_{seg.replace('=', '_').replace(' ', '_')}": res.loc[seg, "auc"]
                            for seg in res.index}
                metrics["val_ndcg10_test_mix"] = res.loc["test-mix weighted", "ndcg@10"]
                auc = res.loc["test-mix weighted", "auc"]
                if auc > best_auc:
                    best_auc, best_epoch, bad = auc, epoch, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
            mlflow.log_metrics(metrics, step=epoch)
            print(f"epoch {epoch}: loss {metrics['train_loss']:.4f}  {metrics['epoch_seconds']:.0f}s"
                  + (f"  val test-mix AUC {auc:.4f}" if early_stop else ""), flush=True)
            if early_stop and bad >= tc["patience"]:
                break

        if not early_stop:
            best_epoch, best_state = epochs, model.state_dict()
        model.load_state_dict(best_state)
        path = MODELS_DIR / f"{run.info.run_id}.pt"
        torch.save({"state_dict": best_state, "cfg": cfg, "setup": setup, "seed": seed, "best_epoch": best_epoch}, path)
        mlflow.log_params({"best_epoch": best_epoch, "model_path": str(path.relative_to(PROJECT_ROOT))})
        if early_stop:
            mlflow.log_metric("best_val_auc_test_mix", best_auc)
    return model, best_epoch, run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=["validation", "test"], default="validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="fixed epochs (required for --setup test)")
    args = parser.parse_args()
    _, best_epoch, run_id = train(args.setup, args.seed, args.epochs)
    print(f"best epoch {best_epoch}; run {run_id}")


if __name__ == "__main__":
    main()
