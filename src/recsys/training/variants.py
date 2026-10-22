"""Model/graph variants for the ablation round (ADR 009). The reference is `Variant()`.

A variant is applied identically to the training graph, the scoring graph, the user
histories and the model, so every ablation is trained and scored consistently.
"""

from dataclasses import asdict, dataclass

import torch
from torch_geometric.data import HeteroData

from recsys.graph.periods import load_period
from recsys.graph.pyg import PYG_DIR

COCLICK = ("article", "coclicked", "article")
MENTIONS = (("article", "mentions", "entity"), ("entity", "mentioned_in", "article"))


@dataclass(frozen=True)
class Variant:
    name: str = "reference"
    drop_edge_types: tuple = ()          # edge types removed from every graph
    drop_node_types: tuple = ()          # node types removed (with all their edges)
    raw_coclick: bool = False            # raw (unpruned, weight = shared users) co-click edges
    degree_weight: bool = True           # 1 / log(1 + degree) on entity -> article edges
    train_graph: str | None = None       # override the training-graph period (A6)
    history_clicks: bool = True          # append training-period clicks to histories (A7)
    user_pooling: str = "mean"           # "mean" or "attention" (A8)
    residual: bool = False               # add projected input text to final article vector (R1)
    drop_all_edges: bool = False         # A0: keep edge types but no edges -> article vector from its own text only
    seen_flag_input: bool = False        # A9: append the article's seen-in-the-graph-period flag to its input features

    def params(self) -> dict:
        return {f"variant.{k}": str(v) for k, v in asdict(self).items()}

    def graph(self, period: str) -> tuple[HeteroData, dict]:
        if self.raw_coclick:
            saved = torch.load(PYG_DIR / "periods" / f"{period}__raw_coclick.pt", weights_only=False)
            data, ids = saved["data"], saved["ids"]
        else:
            data, ids = load_period(period)
        if self.drop_node_types:
            data = data.node_type_subgraph([nt for nt in data.node_types if nt not in self.drop_node_types])
        if self.drop_edge_types:
            data = data.edge_type_subgraph([et for et in data.edge_types if et not in self.drop_edge_types])
        if self.drop_all_edges:
            for et in data.edge_types:
                data[et].edge_index = torch.empty(2, 0, dtype=torch.long)
                if "edge_weight" in data[et]:
                    data[et].edge_weight = torch.empty(0)
        if not self.degree_weight:
            del data["entity", "mentioned_in", "article"].edge_weight
        if self.seen_flag_input:
            # The flag is relative to this graph's own period (9-11 Nov in training, 9-13 Nov at scoring).
            seen = data["article"].seen_in_training.float().unsqueeze(1)
            data["article"].x = torch.cat([data["article"].x, seen], dim=1)
        return data, ids


# Comparison models trained with the same machinery but not judged by the ablation rules.
COMPARISONS = {
    # A0: trained no-graph model (ADR 009): same architecture, loss, training and user encoder.
    "A0": Variant("A0 trained no-graph model", drop_all_edges=True),
}

# ID -> (variant, deciding segments). Deciding segments from ADR 009 (R1 has two).
ABLATIONS = {
    "A1": (Variant("A1 no co-click edges", drop_edge_types=(COCLICK,)), ["article=seen"]),
    "A2": (Variant("A2 no entity edges", drop_edge_types=MENTIONS), ["article=unseen"]),
    # Categories connect to articles only through subcategories, so they go too.
    "A3": (Variant("A3 no subcategory nodes", drop_node_types=("subcategory", "category")), ["article=unseen"]),
    "A4": (Variant("A4 raw co-click edges", raw_coclick=True), ["test-mix weighted"]),
    "A5": (Variant("A5 no degree down-weighting", degree_weight=False), ["test-mix weighted"]),
    "A6": (Variant("A6 no graph/example split", train_graph="validation_score"), ["test-mix weighted"]),
    "A7": (Variant("A7 no training clicks in history", history_clicks=False), ["user=known"]),
    "A8": (Variant("A8 attention pooling", user_pooling="attention"), ["test-mix weighted"]),
    "R1": (Variant("R1 residual text connection", residual=True), ["article=unseen", "test-mix weighted"]),
    # A9 adds an input (the flag was never one): does telling the model which articles are new help?
    "A9": (Variant("A9 seen flag as article input", seen_flag_input=True), ["test-mix weighted"]),
}
