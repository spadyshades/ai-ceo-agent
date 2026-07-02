"""Tool catalog for the agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from src.intelligence import engines
from src.tools import (
    comparator,
    financial_data,
    retriever,
    sentiment,
    source_credibility,
    trend_detector,
    web_search,
)


@dataclass
class ToolSpec:
    name: str
    description: str
    required_params: dict[str, str]
    optional_params: dict[str, str]
    function: Callable[..., Any]
    summary_fn: Callable[[Any], str]


def _summarize_hits(result) -> str:
    if not result:
        return "no results"
    sample = result[0]
    title = getattr(sample, "title", "") or str(sample)[:60]
    return f"{len(result)} results; top: {title[:80]}"


def _summarize_sentiment(result) -> str:
    return f"label={result.label}, score={result.score:.3f}"


def _summarize_strategic_items(result) -> str:
    return f"{len(result)} items"


def _summarize_trend_tuple(result) -> str:
    rising, synthesised = result
    return f"{len(rising)} rising entities, {len(synthesised)} synthesised trends"


def _summarize_score(result) -> str:
    return f"score={result}"


def _summarize_trending(result) -> str:
    return f"{len(result)} trending entities"


def _summarize_financial(result) -> str:
    return result.summary[:120]


_REGISTRY: dict[str, ToolSpec] = {
    "retriever": ToolSpec(
        name="retriever",
        description="Semantic search over the BMW corpus.",
        required_params={"query": "the search string"},
        optional_params={"k": "max results, integer, default 5"},
        function=lambda **kw: retriever.search(**kw),
        summary_fn=_summarize_hits,
    ),
    "hybrid_search": ToolSpec(
        name="hybrid_search",
        description="Combined semantic + keyword search. Better for specific terms, model names, or exact phrases.",
        required_params={"query": "the search string"},
        optional_params={"k": "max results, integer, default 5"},
        function=lambda **kw: retriever.search_hybrid(**kw),
        summary_fn=_summarize_hits,
    ),
    "web_search": ToolSpec(
        name="web_search",
        description="External news search via Google News.",
        required_params={"query": "the search string"},
        optional_params={"limit": "max results, integer, default 10"},
        function=lambda **kw: web_search.search(**kw),
        summary_fn=_summarize_hits,
    ),
    "financial_data": ToolSpec(
        name="financial_data",
        description="Live stock data: price, P/E, 52-week range, market cap, dividend yield. Default ticker: BMW.DE.",
        required_params={},
        optional_params={"ticker": "stock ticker, default BMW.DE"},
        function=lambda **kw: financial_data.get_snapshot(**kw),
        summary_fn=_summarize_financial,
    ),
    "sentiment": ToolSpec(
        name="sentiment",
        description="Classify a piece of text as positive, neutral, or negative using FinBERT.",
        required_params={"text": "text to classify"},
        optional_params={},
        function=lambda text: sentiment.classify(text),
        summary_fn=_summarize_sentiment,
    ),
    "trend_detector": ToolSpec(
        name="trend_detector",
        description="Statistical detection of rising entities in the corpus.",
        required_params={},
        optional_params={
            "label": "entity label, default ORG",
            "days_recent": "integer, default 14",
            "days_baseline": "integer, default 28",
        },
        function=lambda **kw: trend_detector.detect_rising_entities(**kw),
        summary_fn=_summarize_trending,
    ),
    "find_contradicting_evidence": ToolSpec(
        name="find_contradicting_evidence",
        description="Find corpus passages likely to contradict a claim.",
        required_params={"claim": "the claim string to challenge"},
        optional_params={"k": "max results, integer, default 5"},
        function=lambda **kw: comparator.find_contradicting_evidence(**kw),
        summary_fn=_summarize_hits,
    ),
    "source_credibility": ToolSpec(
        name="source_credibility",
        description="Score a known source name (bmw_press, google_news, arxiv, yahoo_finance, hackernews) on a 0.0-1.0 scale.",
        required_params={"source": "source name string"},
        optional_params={"published_at": "ISO datetime string"},
        function=lambda **kw: source_credibility.score_source(**kw),
        summary_fn=_summarize_score,
    ),
    "detect_opportunities": ToolSpec(
        name="detect_opportunities",
        description="Composite engine: identify strategic opportunities. Slow (~60s).",
        required_params={},
        optional_params={},
        function=lambda **kw: engines.detect_opportunities(),
        summary_fn=_summarize_strategic_items,
    ),
    "detect_risks": ToolSpec(
        name="detect_risks",
        description="Composite engine: identify strategic risks. Slow (~60s).",
        required_params={},
        optional_params={},
        function=lambda **kw: engines.detect_risks(),
        summary_fn=_summarize_strategic_items,
    ),
    "detect_trends": ToolSpec(
        name="detect_trends",
        description="Composite engine: identify strategic trends. Slow (~60s).",
        required_params={},
        optional_params={},
        function=lambda **kw: engines.detect_trends(),
        summary_fn=_summarize_trend_tuple,
    ),
}


def list_tools() -> list[str]:
    return list(_REGISTRY.keys())


def get_tool(name: str) -> ToolSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown tool: {name}")
    return _REGISTRY[name]


def format_catalog() -> str:
    lines: list[str] = []
    for spec in _REGISTRY.values():
        lines.append(f"- {spec.name}: {spec.description}")
        if not spec.required_params and not spec.optional_params:
            lines.append("    parameters: none")
            continue
        for p, d in spec.required_params.items():
            lines.append(f'    "{p}" (required): {d}')
        for p, d in spec.optional_params.items():
            lines.append(f'    "{p}" (optional): {d}')
    return "\n".join(lines)


def _canonicalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


_KEY_ALIASES = {
    "searchquery": "query",
    "q": "query",
    "searchstring": "query",
    "input": "text",
    "claimstring": "claim",
    "sourcename": "source",
    "maxresults": "k",
    "numresults": "k",
    "stockticker": "ticker",
    "symbol": "ticker",
}


def _normalize_params(spec: ToolSpec, params: dict[str, Any]) -> dict[str, Any]:
    accepted = set(spec.required_params.keys()) | set(spec.optional_params.keys())
    if not accepted:
        return {}
    normalized: dict[str, Any] = {}
    for raw_key, value in params.items():
        if raw_key in accepted:
            normalized[raw_key] = value
            continue
        canonical = _canonicalize_key(raw_key)
        for ak in accepted:
            if _canonicalize_key(ak) == canonical:
                normalized[ak] = value
                break
        else:
            alias_target = _KEY_ALIASES.get(canonical)
            if alias_target and alias_target in accepted:
                normalized[alias_target] = value
    return normalized


def execute(name: str, params: dict[str, Any]) -> Any:
    spec = get_tool(name)
    params = _normalize_params(spec, params or {})
    missing = [p for p in spec.required_params if p not in params]
    if missing:
        raise ValueError(
            f"missing required parameter(s) for {name}: {', '.join(missing)}"
        )
    return spec.function(**params)


def summarize(name: str, result: Any) -> str:
    try:
        return get_tool(name).summary_fn(result)
    except Exception as exc:
        return f"summary failed: {exc}"
