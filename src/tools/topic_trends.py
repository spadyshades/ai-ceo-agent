"""TF-IDF topic trend detection across time windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sklearn.feature_extraction.text import TfidfVectorizer

from src.processing.indexer import get_collection


@dataclass
class TopicTrend:
    term: str
    recent_tfidf: float
    baseline_tfidf: float
    growth_rate: float


def detect_rising_topics(
    days_recent: int = 14,
    days_baseline: int = 28,
    top_n: int = 20,
    max_features: int = 5000,
    min_df: int = 2,
) -> list[TopicTrend]:
    """Compare TF-IDF term importance between recent and baseline windows."""
    now = datetime.now(timezone.utc)
    recent_cutoff = (now - timedelta(days=days_recent)).isoformat()
    baseline_cutoff = (now - timedelta(days=days_baseline)).isoformat()

    collection = get_collection()
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents", [])
    metas = all_data.get("metadatas", [])

    recent_docs: list[str] = []
    baseline_docs: list[str] = []

    for doc, meta in zip(docs, metas):
        published = (meta or {}).get("published_at", "")
        if not published or published < baseline_cutoff:
            continue
        if published >= recent_cutoff:
            recent_docs.append(doc)
        else:
            baseline_docs.append(doc)

    if not recent_docs:
        return []

    # Build TF-IDF on the combined corpus, then split scores
    all_texts = recent_docs + baseline_docs
    labels = ["recent"] * len(recent_docs) + ["baseline"] * len(baseline_docs)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    terms = vectorizer.get_feature_names_out()

    recent_mask = [i for i, l in enumerate(labels) if l == "recent"]
    baseline_mask = [i for i, l in enumerate(labels) if l == "baseline"]

    recent_mean = tfidf_matrix[recent_mask].mean(axis=0).A1 if recent_mask else [0] * len(terms)
    baseline_mean = tfidf_matrix[baseline_mask].mean(axis=0).A1 if baseline_mask else [0] * len(terms)

    trends: list[TopicTrend] = []
    for i, term in enumerate(terms):
        r_score = float(recent_mean[i])
        b_score = float(baseline_mean[i])
        if r_score < 0.01:
            continue
        if b_score > 0:
            growth = r_score / b_score
        else:
            growth = float("inf") if r_score > 0 else 0.0

        if growth > 1.2:
            trends.append(TopicTrend(
                term=term,
                recent_tfidf=round(r_score, 4),
                baseline_tfidf=round(b_score, 4),
                growth_rate=round(growth, 2) if growth != float("inf") else float("inf"),
            ))

    trends.sort(
        key=lambda t: (t.growth_rate if t.growth_rate != float("inf") else 1e9),
        reverse=True,
    )
    return trends[:top_n]
