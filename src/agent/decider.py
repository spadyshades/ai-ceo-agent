"""Decider node: produce ranked strategic recommendations."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.agent.schema import AgentState, Recommendation
from src.utils.llm import complete


logger = logging.getLogger(__name__)


_PROMPT = """You are the decision component of a strategic intelligence agent for BMW.

User goal: {goal}

Analysis summary: {analysis_summary}

Key findings:
{key_findings}

Opportunities ({n_opp}):
{opp_summary}

Risks ({n_risks}):
{risk_summary}

Trends ({n_trends}):
{trend_summary}

Retrieved evidence chunks (cite these IDs in your recommendations):
{retrieved_summary}

Produce 3 to 5 strategic recommendations for the CEO. Each must be specific,
actionable, and grounded in the evidence above.

Each recommendation MUST include:
- title (max 12 words)
- rationale (2-3 sentences)
- priority ("High", "Medium", or "Low")
- expected_impact (one sentence)
- risk_assessment (one sentence)
- confidence (number 0.0 to 1.0)
- evidence_chunk_ids: list of at least 3 chunk IDs from above, from at least 2 different sources

Return ONLY a JSON array of recommendation objects. No commentary.

JSON array:"""


def _format_items(items, max_items: int = 8) -> str:
    if not items:
        return "(none)"
    lines = []
    for it in items[:max_items]:
        chunk_str = ",".join(it.evidence_chunk_ids[:5])
        lines.append(
            f"- [{it.impact}] {it.title}: {it.description[:180]} "
            f"(evidence: {chunk_str})"
        )
    return "\n".join(lines)


def _format_retrieved(retrieved_chunks, limit: int = 40) -> str:
    if not retrieved_chunks:
        return "(none)"
    return "\n".join(
        f"- {ref.chunk_id} [{ref.source}] {ref.title[:90]}"
        for ref in retrieved_chunks[:limit]
    )


def _extract_json_array(text: str) -> list[dict] | None:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def _build_recommendation(
    raw: dict,
    sources_by_chunk: dict[str, str],
) -> Recommendation | None:
    try:
        evidence_ids = raw.get("evidence_chunk_ids") or []
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        evidence_ids = [str(c) for c in evidence_ids]
        sources = sorted({
            sources_by_chunk[cid]
            for cid in evidence_ids
            if cid in sources_by_chunk and sources_by_chunk[cid]
        })
        return Recommendation(
            title=str(raw.get("title", "")).strip(),
            rationale=str(raw.get("rationale", "")).strip(),
            priority=str(raw.get("priority", "Medium")).strip().capitalize(),
            expected_impact=str(raw.get("expected_impact", "")).strip(),
            risk_assessment=str(raw.get("risk_assessment", "")).strip(),
            confidence=float(raw.get("confidence", 0.5)),
            evidence_chunk_ids=evidence_ids,
            evidence_sources=sources,
        )
    except Exception as exc:
        logger.warning("Recommendation parse error: %s", exc)
        return None


def decide_node(state: AgentState) -> dict[str, Any]:
    logger.info("Decider: synthesising recommendations")

    # Source lookup spans intelligence items AND retrieved chunks
    sources_by_chunk: dict[str, str] = {}
    for ref in state.retrieved_chunks:
        if ref.source:
            sources_by_chunk[ref.chunk_id] = ref.source
    for collection in (state.opportunities, state.risks, state.trends):
        for item in collection:
            for cid, src in zip(item.evidence_chunk_ids, item.evidence_sources):
                if src:
                    sources_by_chunk[cid] = src

    analysis_summary = (
        state.analysis.summary if state.analysis else "(no analysis available)"
    )
    key_findings = (
        "\n".join(f"- {f}" for f in state.analysis.key_findings)
        if state.analysis and state.analysis.key_findings
        else "(none)"
    )

    prompt = _PROMPT.format(
        goal=state.goal,
        analysis_summary=analysis_summary,
        key_findings=key_findings,
        n_opp=len(state.opportunities),
        n_risks=len(state.risks),
        n_trends=len(state.trends),
        opp_summary=_format_items(state.opportunities),
        risk_summary=_format_items(state.risks),
        trend_summary=_format_items(state.trends),
        retrieved_summary=_format_retrieved(state.retrieved_chunks),
    )
    response = complete(prompt, temperature=0.3)
    raw_items = _extract_json_array(response) or []

    recommendations: list[Recommendation] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        rec = _build_recommendation(raw, sources_by_chunk)
        if rec and rec.title:
            recommendations.append(rec)

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recommendations.sort(
        key=lambda r: (priority_order.get(r.priority, 1), -r.confidence)
    )

    logger.info("Decider: %d recommendations", len(recommendations))
    return {"recommendations": recommendations}
