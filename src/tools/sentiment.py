"""Document-level sentiment classification using FinBERT."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from transformers import pipeline


logger = logging.getLogger(__name__)


_MODEL_NAME = "ProsusAI/finbert"


@dataclass
class SentimentResult:
    label: str
    score: float


@lru_cache(maxsize=1)
def _get_pipeline():
    logger.info("Loading sentiment model: %s", _MODEL_NAME)
    return pipeline(
        "sentiment-analysis",
        model=_MODEL_NAME,
        truncation=True,
        max_length=512,
    )


def classify(text: str) -> SentimentResult:
    if not text or not text.strip():
        return SentimentResult(label="neutral", score=0.0)
    clf = _get_pipeline()
    result = clf(text[:2000])[0]
    label = result["label"].lower()
    return SentimentResult(label=label, score=float(result["score"]))


def classify_batch(texts: list[str]) -> list[SentimentResult]:
    if not texts:
        return []
    clf = _get_pipeline()
    truncated = [t[:2000] if t and t.strip() else "" for t in texts]
    raw_results = clf(truncated)
    return [
        SentimentResult(
            label=r["label"].lower(),
            score=float(r["score"]),
        )
        for r in raw_results
    ]
