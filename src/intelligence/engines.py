"""Strategic intelligence engines: opportunities, risks, trends."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.tools.retriever import RetrievalHit, search
from src.tools.sentiment import classify_batch
from src.tools.source_credibility import score_source
from src.tools.trend_detector import TrendingEntity, detect_rising_entities
from src.utils.llm import complete


logger = logging.getLogger(__name__)


@dataclass
class StrategicItem:
    title: str
    description: str
    impact: str
    confidence: float
    evidence_chunk_ids: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)


_OPPORTUNITY_QUERIES = [
    "BMW new partnership announcement",
    "BMW investment in emerging technology",
    "BMW expansion into new market",
    "BMW product launch innovation",
    "BMW electric vehicle growth",
]

_RISK_QUERIES = [
    "BMW supply chain disruption",
    "BMW regulatory penalty fine",
    "BMW competitive threat market share",
    "BMW recall quality issue",
    "BMW China sales decline",
]

_OPPORTUNITY_PROMPT = (
    "You are a strategic analyst preparing intelligence for the CEO of BMW.\n"
    "Below are excerpts from recent news, press releases, and research.\n"
    "Identify up to 5 distinct STRATEGIC OPPORTUNITIES for BMW supported by these excerpts.\n\n"
    "For each opportunity, output an object with these fields:\n"
    '  "title": short headline (max 12 words)\n'
    '  "description": 2-3 sentences explaining the opportunity\n'
    '  "impact": one of "Low", "Medium", "High"\n'
    '  "confidence": a number from 0.0 to 1.0\n'
    '  "evidence_chunk_ids": list of chunk IDs from the excerpts that support it\n\n'
    "Return a single valid JSON array with double-quoted property names and no trailing commas. "
    "No surrounding text or commentary.\n\n"
    "Excerpts:\n{excerpts}\n\n"
    "JSON array:"
)

_RISK_PROMPT = (
    "You are a strategic analyst preparing intelligence for the CEO of BMW.\n"
    "Below are excerpts from recent news, press releases, and research.\n"
    "Identify up to 5 distinct STRATEGIC RISKS facing BMW supported by these excerpts.\n\n"
    "For each risk, output an object with these fields:\n"
    '  "title": short headline (max 12 words)\n'
    '  "description": 2-3 sentences explaining the risk\n'
    '  "impact": one of "Low", "Medium", "High"\n'
    '  "confidence": a number from 0.0 to 1.0\n'
    '  "evidence_chunk_ids": list of chunk IDs from the excerpts that support it\n\n'
    "Return a single valid JSON array with double-quoted property names and no trailing commas. "
    "No surrounding text or commentary.\n\n"
    "Excerpts:\n{excerpts}\n\n"
    "JSON array:"
)

_TREND_PROMPT = (
    "You are a strategic analyst preparing intelligence for the CEO of BMW.\n"
    "Below is a list of entities with rising mention frequency in recent coverage,\n"
    "and supporting excerpts from the corpus.\n"
    "Identify up to 5 distinct STRATEGIC TRENDS that BMW management should monitor.\n\n"
    "For each trend, output an object with these fields:\n"
    '  "title": short headline (max 12 words)\n'
    '  "description": 2-3 sentences explaining the trend and why it matters\n'
    '  "impact": one of "Low", "Medium", "High"\n'
    '  "confidence": a number from 0.0 to 1.0\n'
    '  "evidence_chunk_ids": list of chunk IDs from the excerpts that support it\n\n'
    "Return a single valid JSON array with double-quoted property names and no trailing commas. "
    "No surrounding text or commentary.\n\n"
    "Rising entities:\n{rising}\n\n"
    "Excerpts:\n{excerpts}\n\n"
    "JSON array:"
)


def _gather_evidence(queries: list[str], per_query: int = 3) -> list[RetrievalHit]:
    seen: set[str] = set()
    unique: list[RetrievalHit] = []
    for query in queries:
        for hit in search(query, k=per_query):
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                unique.append(hit)
    return unique


def _format_excerpts(hits: list[RetrievalHit], max_hits: int = 15) -> str:
    blocks = []
    for hit in hits[:max_hits]:
        snippet = hit.text[:500].replace("\n", " ")
        published = hit.published_at[:10] if hit.published_at else "unknown"
        blocks.append(
            f"[{hit.chunk_id}] ({hit.source}, {published}) {hit.title}\n{snippet}"
        )
    return "\n\n".join(blocks)


def _clean_json_string(s: str) -> str:
    """Best-effort cleanup of common LLM JSON mistakes."""
    # Quote unquoted property names: {key: ...} or , key: ...
    s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', s)
    # Remove trailing commas before closing braces/brackets
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array with multiple fallback strategies."""
    # 1. Direct parse of array region
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        raw = match.group(0)
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
        try:
            data = json.loads(_clean_json_string(raw))
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass

    # 2. Extract individual JSON objects
    objects: list[dict[str, Any]] = []
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        chunk = m.group(0)
        for candidate in (chunk, _clean_json_string(chunk)):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    objects.append(obj)
                    break
            except json.JSONDecodeError:
                continue

    if not objects:
        logger.warning("No JSON parseable in LLM response")
    return objects


def _to_strategic_items(
    raw_items: list[dict[str, Any]],
    hits: list[RetrievalHit],
) -> list[StrategicItem]:
    chunk_to_source = {hit.chunk_id: hit.source for hit in hits}
    items: list[StrategicItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        evidence_ids = raw.get("evidence_chunk_ids") or []
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        sources = sorted({
            chunk_to_source[cid]
            for cid in evidence_ids
            if cid in chunk_to_source
        })
        try:
            items.append(
                StrategicItem(
                    title=str(raw.get("title", "")).strip(),
                    description=str(raw.get("description", "")).strip(),
                    impact=str(raw.get("impact", "Medium")).strip().capitalize(),
                    confidence=float(raw.get("confidence", 0.5)),
                    evidence_chunk_ids=[str(cid) for cid in evidence_ids],
                    evidence_sources=sources,
                )
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping malformed item: %s", exc)
    return items


def detect_opportunities() -> list[StrategicItem]:
    hits = _gather_evidence(_OPPORTUNITY_QUERIES)
    if not hits:
        return []
    response = complete(
        _OPPORTUNITY_PROMPT.format(excerpts=_format_excerpts(hits)), temperature=0.3
    )
    return _to_strategic_items(_extract_json_array(response), hits)


def detect_risks() -> list[StrategicItem]:
    hits = _gather_evidence(_RISK_QUERIES)
    if not hits:
        return []

    sentiments = classify_batch([hit.text[:1000] for hit in hits])
    weighted = list(zip(hits, sentiments))
    weighted.sort(
        key=lambda pair: (
            pair[1].label == "negative",
            pair[1].score if pair[1].label == "negative" else 0.0,
        ),
        reverse=True,
    )
    ordered_hits = [pair[0] for pair in weighted]

    response = complete(
        _RISK_PROMPT.format(excerpts=_format_excerpts(ordered_hits)), temperature=0.3
    )
    return _to_strategic_items(_extract_json_array(response), ordered_hits)


def detect_trends(top_entities: int = 10) -> tuple[list[TrendingEntity], list[StrategicItem]]:
    rising = detect_rising_entities(label="ORG", top_n=top_entities)
    if not rising:
        return [], []

    rising_text = "\n".join(
        f"- {r.entity} (recent={r.recent_count}, baseline={r.baseline_count}, "
        f"growth={r.growth_rate:.2f})"
        for r in rising
    )

    seed_queries = [f"BMW {r.entity}" for r in rising[:5]]
    hits = _gather_evidence(seed_queries, per_query=2)
    excerpts = _format_excerpts(hits) if hits else "No supporting excerpts retrieved."

    response = complete(
        _TREND_PROMPT.format(rising=rising_text, excerpts=excerpts),
        temperature=0.3,
    )
    items = _to_strategic_items(_extract_json_array(response), hits)
    return rising, items


def score_items_by_credibility(items: list[StrategicItem]) -> list[StrategicItem]:
    for item in items:
        if not item.evidence_sources:
            continue
        weights = [score_source(src) for src in item.evidence_sources]
        if weights:
            avg = sum(weights) / len(weights)
            item.confidence = round(item.confidence * avg, 3)
    return items
