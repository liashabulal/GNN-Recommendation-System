"""Non-graph baselines for MIND: random, popularity and TF-IDF content similarity.

Every model follows the same two-step interface:

    model.fit(train_behaviors, news)   # learn only from the split's training period
    model.score(target_behaviors)      # -> DataFrame[impression_id, news_id, score]

`news` is the full article catalogue (text and metadata). Article content is available
the moment an article is published, so using it for unseen articles is not leakage;
anything derived from *clicks* must come from `train_behaviors` only.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.data.mind import explode_impressions


def _candidate_pairs(target: pd.DataFrame) -> pd.DataFrame:
    """One row per (impression, candidate) to be scored, with the impression's user."""
    return explode_impressions(target)[["impression_id", "user_id", "news_id"]]


class RandomScores:
    """Uniform random scores: the sanity floor (expected AUC 0.5)."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, train: pd.DataFrame, news: pd.DataFrame) -> "RandomScores":
        return self

    def score(self, target: pd.DataFrame) -> pd.DataFrame:
        pairs = _candidate_pairs(target)
        rng = np.random.default_rng(self.seed)
        return pairs[["impression_id", "news_id"]].assign(score=rng.random(len(pairs)))


class Popularity:
    """Score = number of clicks the article received in training-period impressions.

    Articles never clicked in the training period (including all unseen ones) score 0.
    The same ranking is shown to every user.
    """

    name = "popularity"

    def fit(self, train: pd.DataFrame, news: pd.DataFrame) -> "Popularity":
        clicks = explode_impressions(train)
        self.click_counts_ = clicks.loc[clicks["clicked"] == 1, "news_id"].value_counts()
        return self

    def score(self, target: pd.DataFrame) -> pd.DataFrame:
        pairs = _candidate_pairs(target)
        scores = pairs["news_id"].map(self.click_counts_).fillna(0).astype(float)
        return pairs[["impression_id", "news_id"]].assign(score=scores)


class TfidfContent:
    """Content-based: cosine similarity between a user profile and each candidate.

    - Article vector: TF-IDF of title + abstract. The vocabulary and IDF weights are fitted
      only on articles seen in the training period, then applied to every article.
    - User profile: mean of the user's history article vectors (from the target
      impression's history snapshot, so new users are covered). Empty history -> zero
      profile -> all candidates tie.
    - Score: cosine(profile, article).
    """

    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (1, 1), sublinear_tf: bool = False, min_df: int = 2):
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df

    def fit(self, train: pd.DataFrame, news: pd.DataFrame) -> "TfidfContent":
        text = news["title"].fillna("") + " " + news["abstract"].fillna("")
        seen = set(train["history"].explode().dropna()) | set(train["candidates"].explode())

        self.vectorizer_ = TfidfVectorizer(
            stop_words="english",
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
        ).fit(text[news["news_id"].isin(seen)])

        # Rows are L2-normalised by the vectorizer, so a dot product is a cosine.
        self.article_vectors_ = self.vectorizer_.transform(text)
        self.row_of_ = pd.Series(np.arange(len(news)), index=news["news_id"].to_numpy())
        return self

    def _user_profiles(self, users: pd.DataFrame) -> sp.csr_matrix:
        """L2-normalised mean history vector per row of `users` (columns user_id, history)."""
        lengths = users["history"].map(len).to_numpy()
        rows = np.repeat(np.arange(len(users)), lengths)
        cols = self.row_of_.loc[users["history"].explode().dropna()].to_numpy()
        weights = np.repeat(1 / np.maximum(lengths, 1), lengths)
        averaging = sp.csr_matrix((weights, (rows, cols)), shape=(len(users), len(self.row_of_)))
        return normalize(averaging @ self.article_vectors_)

    def score(self, target: pd.DataFrame, user_chunk: int = 2000) -> pd.DataFrame:
        # A user's history is the same snapshot on all their impressions, so one profile per user.
        users = target.drop_duplicates("user_id")[["user_id", "history"]].reset_index(drop=True)
        assert target["history"].map(tuple).groupby(target["user_id"]).nunique().eq(1).all()
        profiles = self._user_profiles(users)

        pairs = _candidate_pairs(target)
        user_pos = pd.Series(users.index, index=users["user_id"]).loc[pairs["user_id"]].to_numpy()
        cand_ids = pairs["news_id"].unique()
        cand_pos = pd.Series(np.arange(len(cand_ids)), index=cand_ids).loc[pairs["news_id"]].to_numpy()
        cand_vectors = self.article_vectors_[self.row_of_.loc[cand_ids].to_numpy()]

        # Dense (users x candidate articles) similarity, a chunk of users at a time,
        # then pick out the pairs that were actually shown.
        scores = np.empty(len(pairs))
        for start in range(0, len(users), user_chunk):
            sims = (profiles[start : start + user_chunk] @ cand_vectors.T).toarray()
            in_chunk = (user_pos >= start) & (user_pos < start + user_chunk)
            scores[in_chunk] = sims[user_pos[in_chunk] - start, cand_pos[in_chunk]]
        return pairs[["impression_id", "news_id"]].assign(score=scores)


class EmbeddingContent:
    """Frozen sentence embeddings, no graph (ADR 009's "embedding baseline").

    - Article vector: the frozen MiniLM text embedding (ADR 006), unit length.
    - User vector: mean of the user's history article embeddings, using the same history
      rule as the GNN (ADR 007): last 50 of the history column + training-period clicks
      before the impression time, each article once.
    - Score: cosine(user vector, article vector). Empty history -> all candidates tie.
    """

    name = "embedding"

    def __init__(self, max_history: int = 50):
        self.max_history = max_history

    def fit(self, train: pd.DataFrame, news: pd.DataFrame) -> "EmbeddingContent":
        from recsys.features.build import FEATURES_DIR

        self.train_ = train  # only its clicks are used, to extend histories
        self.vectors_ = np.load(FEATURES_DIR / "article_text.npy")
        self.row_of_ = pd.Series(np.arange(len(self.vectors_)),
                                 index=pd.read_parquet(FEATURES_DIR / "article_ids.parquet")["news_id"])
        return self

    def score(self, target: pd.DataFrame) -> pd.DataFrame:
        from recsys.models.history import history_tensor, impression_histories

        histories = impression_histories(target, self.train_, self.max_history)
        idx = history_tensor(histories, self.row_of_, self.max_history).numpy()
        mask = idx >= 0
        users = (self.vectors_[np.maximum(idx, 0)] * mask[..., None]).sum(axis=1)
        users = normalize(users)  # zero rows (empty history) stay zero
        user_row = pd.Series(np.arange(len(histories)), index=histories.index)

        pairs = _candidate_pairs(target)
        u = users[user_row.loc[pairs["impression_id"]].to_numpy()]
        a = self.vectors_[self.row_of_.loc[pairs["news_id"]].to_numpy()]
        return pairs[["impression_id", "news_id"]].assign(score=(u * a).sum(axis=1))
