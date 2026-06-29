"""Named entity extraction with spaCy."""

from __future__ import annotations

import logging
from functools import lru_cache

import spacy
from spacy.language import Language


logger = logging.getLogger(__name__)


_RELEVANT_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "MONEY", "DATE", "EVENT"}


@lru_cache(maxsize=1)
def _load_model() -> Language:
    try:
        return spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


def extract_entities(text: str, max_chars: int = 50000) -> dict[str, list[str]]:
    """Extract named entities grouped by label.

    Text is truncated to ``max_chars`` to bound latency. The front of each
    document carries the most informative entities, so this rarely loses signal.
    """
    nlp = _load_model()
    doc = nlp(text[:max_chars])

    grouped: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for ent in doc.ents:
        if ent.label_ not in _RELEVANT_LABELS:
            continue
        text_norm = ent.text.strip()
        if not text_norm:
            continue
        grouped.setdefault(ent.label_, [])
        seen.setdefault(ent.label_, set())
        if text_norm not in seen[ent.label_]:
            grouped[ent.label_].append(text_norm)
            seen[ent.label_].add(text_norm)
    return grouped
