"""Score a source by tier and freshness."""

from __future__ import annotations

from datetime import datetime, timezone


_SOURCE_TIER = {
    "bmw_press": 1,
    "arxiv": 1,
    "google_news": 2,
    "yahoo_finance": 2,
    "hackernews": 3,
}

_TIER_SCORE = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.3}


def get_source_tier(source: str) -> int:
    """Return the tier of a known source (1 highest, 4 unknown)."""
    return _SOURCE_TIER.get(source, 4)


def score_source(source: str, published_at: str | datetime | None = None) -> float:
    """Return a credibility score from 0.0 to 1.0.

    The score combines a fixed tier weight with a freshness decay; a tier-1
    source published today scores 1.0, the same source one year old scores
    around 0.4.
    """
    tier = _SOURCE_TIER.get(source, 4)
    base = _TIER_SCORE[tier]

    if not published_at:
        return round(base * 0.7, 3)

    if isinstance(published_at, str):
        try:
            published_at = datetime.fromisoformat(
                published_at.replace("Z", "+00:00")
            )
        except ValueError:
            return round(base * 0.7, 3)

    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    age_days = max(0, (datetime.now(timezone.utc) - published_at).days)
    freshness = max(0.4, 1.0 - age_days / 365.0)
    return round(base * freshness, 3)
