"""Compute starting node features once. Run with `uv run python -m recsys.features.build`.

Outputs in data/processed/features/ (see ADR 006):

    article_text.npy       float32 [n_articles, 384]   frozen sentence embedding of title + abstract
    article_ids.parquet    row order of article_text.npy (all articles in the MIND catalogue)
    entity_vectors.npy     float32 [n_entities, 100]   MIND entity vector, zeros if missing
    entity_ids.parquet     row order + `missing` flag (1 = no MIND vector)
    <setup>/subcategory_text.npy, subcategory_ids.parquet
    <setup>/category_text.npy,    category_ids.parquet
                           mean article text embedding over the setup's articles
    metadata.json          encoder name, versions, shapes

Article and entity features are content-only, so one file serves both setups.
Subcategory and category features are per setup, because they average that setup's articles.
Entity-type features are learnable and therefore not computed here.
"""

import json
import tomllib

import numpy as np
import pandas as pd

from recsys.graph.build import load_graph
from recsys.paths import CONFIGS_DIR, PROCESSED_DIR, mind_split_dir

FEATURES_DIR = PROCESSED_DIR / "features"
SETUPS = ("validation", "test")


def load_model_config() -> dict:
    with open(CONFIGS_DIR / "model.toml", "rb") as f:
        return tomllib.load(f)


def article_texts(articles: pd.DataFrame, news: pd.DataFrame) -> list[str]:
    text = news.set_index("news_id").reindex(articles["news_id"])
    return (text["title"].fillna("") + " " + text["abstract"].fillna("")).str.strip().tolist()


def encode_articles(texts: list[str], cfg: dict) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(cfg["model"], device=device)
    emb = model.encode(texts, batch_size=cfg["batch_size"], normalize_embeddings=cfg["normalize"],
                       convert_to_numpy=True, show_progress_bar=True)
    assert emb.shape[1] == cfg["dim"], f"encoder returned {emb.shape[1]} dims, config says {cfg['dim']}"
    return emb.astype(np.float32)


def mind_entity_vectors() -> dict[str, np.ndarray]:
    """Union of both splits' entity_embedding.vec (vectors for shared IDs are identical)."""
    vectors = {}
    for split in ("train", "dev"):
        with open(mind_split_dir(split) / "entity_embedding.vec") as f:
            for line in f:
                parts = line.rstrip().split("\t")
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def entity_features(entity_ids: list[str], vectors: dict, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """MIND vector per entity; zeros and missing = 1 where there is none (option A, ADR 006)."""
    out = np.zeros((len(entity_ids), dim), dtype=np.float32)
    missing = np.ones(len(entity_ids), dtype=np.int8)
    for i, q in enumerate(entity_ids):
        if q in vectors:
            out[i], missing[i] = vectors[q], 0
    return out, missing


def group_mean(article_emb: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """Mean embedding of the articles in each group (rows: group id)."""
    return article_emb.groupby(keys.to_numpy()).mean()


def main() -> None:
    from recsys.experiments.run_baselines import load_catalogue

    cfg = load_model_config()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    news = load_catalogue()

    # Articles: every article in the catalogue (the test setup's article set is the full catalogue).
    articles = news[["news_id"]].sort_values("news_id").reset_index(drop=True)
    emb = encode_articles(article_texts(articles, news), cfg["text_encoder"])
    np.save(FEATURES_DIR / "article_text.npy", emb)
    articles.to_parquet(FEATURES_DIR / "article_ids.parquet", index=False)
    article_emb = pd.DataFrame(emb, index=articles["news_id"])
    print(f"article_text: {emb.shape}")

    # Entities: all entities of the test setup (a superset of the validation setup's).
    entity_ids = sorted(load_graph("test")["nodes_entity"]["entity_id"])
    ent, missing = entity_features(entity_ids, mind_entity_vectors(), cfg["features"]["entity_vector_dim"])
    np.save(FEATURES_DIR / "entity_vectors.npy", ent)
    pd.DataFrame({"entity_id": entity_ids, "missing": missing}).to_parquet(FEATURES_DIR / "entity_ids.parquet",
                                                                          index=False)
    print(f"entity_vectors: {ent.shape}, missing: {missing.sum():,}")

    # Subcategories and categories: mean article embedding within each setup.
    for setup in SETUPS:
        nodes = load_graph(setup)["nodes_article"].set_index("news_id")
        setup_emb = article_emb.loc[nodes.index]
        out = FEATURES_DIR / setup
        out.mkdir(exist_ok=True)
        for name, keys in (("subcategory", nodes["subcategory_id"]), ("category", nodes["category"])):
            means = group_mean(setup_emb, keys)
            np.save(out / f"{name}_text.npy", means.to_numpy(dtype=np.float32))
            pd.DataFrame({f"{name}_id": means.index}).to_parquet(out / f"{name}_ids.parquet", index=False)
            print(f"{setup} {name}_text: {means.shape}")

    import sentence_transformers
    (FEATURES_DIR / "metadata.json").write_text(json.dumps({
        "text_encoder": cfg["text_encoder"],
        "sentence_transformers_version": sentence_transformers.__version__,
        "article_text_shape": list(emb.shape),
        "entity_vectors_shape": list(ent.shape),
        "entities_missing_vector": int(missing.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
