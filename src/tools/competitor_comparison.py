"""Competitor comparison tool: mention counts, sentiment, and evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.processing.indexer import get_collection
from src.tools.sentiment import classify_batch


@dataclass
class CompetitorProfile:
    name: str
    mention_count: int
    sentiment_positive: int = 0
    sentiment_neutral: int = 0
    sentiment_negative: int = 0
    avg_sentiment_score: float = 0.0
    sample_titles: list[str] = field(default_factory=list)


def compare(
    competitors: list[str] | None = None,
    max_samples: int = 5,
) -> list[CompetitorProfile]:
    """Compare mention frequency and sentiment across competitors."""
    if competitors is None:
        from src.config import COMPETITORS
        competitors = COMPETITORS

    collection = get_collection()
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents", [])
    metas = all_data.get("metadatas", [])

    profiles: list[CompetitorProfile] = []
    for comp in competitors:
        needle = comp.lower()
        matching_texts: list[str] = []
        matching_titles: list[str] = []

        for doc, meta in zip(docs, metas):
            if needle in (doc or "").lower():
                matching_texts.append(doc[:512])
                title = meta.get("title", "") if meta else ""
                if title and title not in matching_titles:
                    matching_titles.append(title)

        profile = CompetitorProfile(
            name=comp,
            mention_count=len(matching_texts),
            sample_titles=matching_titles[:max_samples],
        )

        if matching_texts:
            sentiments = classify_batch(matching_texts[:50])
            for s in sentiments:
                if s.label == "positive":
                    profile.sentiment_positive += 1
                elif s.label == "negative":
                    profile.sentiment_negative += 1
                else:
                    profile.sentiment_neutral += 1
            total = len(sentiments)
            if total:
                score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
                profile.avg_sentiment_score = round(
                    sum(score_map.get(s.label, 0) for s in sentiments) / total, 3
                )

        profiles.append(profile)

    profiles.sort(key=lambda p: p.mention_count, reverse=True)
    return profiles
