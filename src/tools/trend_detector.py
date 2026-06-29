"""Detect rising entities by comparing time windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.processing.indexer import get_collection


@dataclass
class TrendingEntity:
    entity: str
    label: str
    recent_count: int
    baseline_count: int
    growth_rate: float


def _parse_entities(meta: dict, label: str) -> list[str]:
    field = f"entities_{label.lower()}"
    raw = meta.get(field, "")
    if not raw:
        return []
    return [e.strip() for e in raw.split(";") if e.strip()]


def detect_rising_entities(
    label: str = "ORG",
    days_recent: int = 14,
    days_baseline: int = 28,
    min_count: int = 2,
    top_n: int = 20,
) -> list[TrendingEntity]:
    """Compare entity mention frequencies between recent and baseline windows.

    Args:
        label: spaCy entity label (ORG, PERSON, GPE, PRODUCT, ...).
        days_recent: Window size for the recent window, ending now.
        days_baseline: Total window depth; baseline covers
            (days_baseline - days_recent) days immediately before the recent window.
        min_count: Minimum recent occurrences to be considered trending.
        top_n: Return the top N entities by growth rate.
    """
    now = datetime.now(timezone.utc)
    recent_cutoff = (now - timedelta(days=days_recent)).isoformat()
    baseline_cutoff = (now - timedelta(days=days_baseline)).isoformat()

    collection = get_collection()
    all_chunks = collection.get(include=["metadatas"])

    recent_counts: dict[str, int] = {}
    baseline_counts: dict[str, int] = {}

    for meta in all_chunks.get("metadatas", []) or []:
        published = meta.get("published_at", "")
        if not published or published < baseline_cutoff:
            continue
        entities = _parse_entities(meta, label)
        if not entities:
            continue
        target = recent_counts if published >= recent_cutoff else baseline_counts
        for ent in set(entities):
            target[ent] = target.get(ent, 0) + 1

    baseline_window_days = max(1, days_baseline - days_recent)
    trending: list[TrendingEntity] = []
    for entity, recent_count in recent_counts.items():
        if recent_count < min_count:
            continue
        baseline_count = baseline_counts.get(entity, 0)
        recent_rate = recent_count / days_recent
        baseline_rate = baseline_count / baseline_window_days
        if baseline_rate == 0:
            growth_rate = float("inf") if recent_count > 0 else 0.0
        else:
            growth_rate = recent_rate / baseline_rate
        trending.append(
            TrendingEntity(
                entity=entity,
                label=label.upper(),
                recent_count=recent_count,
                baseline_count=baseline_count,
                growth_rate=growth_rate,
            )
        )

    trending.sort(
        key=lambda t: (t.growth_rate if t.growth_rate != float("inf") else 1e9),
        reverse=True,
    )
    return trending[:top_n]
