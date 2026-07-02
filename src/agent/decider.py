"""Decider node: produce ranked strategic recommendations."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.schema import AgentState, Recommendation
from src.utils.llm import complete_json


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

Return a JSON object with a single key "recommendations" containing an array.
Each recommendation has:
- "title" (max 12 words)
- "rationale" (2-3 sentences)
- "priority": "High", "Medium", or "Low"
- "expected_impact" (one sentence)
- "risk_assessment" (one sentence)
- "confidence" (number 0.0 to 1.0)
- "evidence_chunk_ids": list of at least 3 chunk IDs from the retrieved evidence above

JSON:"""


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


def _build_source_map(state: AgentState) -> dict[str, str]:
    sources: dict[str, str] = {}
    for ref in state.retrieved_chunks:
        if ref.source:
            sources[ref.chunk_id] = ref.source
    for collection in (state.opportunities, state.risks, state.trends):
        for item in collection:
            for cid, src in zip(item.evidence_chunk_ids, item.evidence_sources):
                if src:
                    sources[cid] = src
    return sources


def _enrich_evidence(
    rec: Recommendation,
    sources_by_chunk: dict[str, str],
    state: AgentState,
) -> Recommendation:
    resolved_sources = sorted({
        sources_by_chunk[cid]
        for cid in rec.evidence_chunk_ids
        if cid in sources_by_chunk and sources_by_chunk[cid]
    })

    if len(resolved_sources) < 2 and state.retrieved_chunks:
        used_ids = set(rec.evidence_chunk_ids)
        for ref in state.retrieved_chunks:
            if ref.source and ref.source not in resolved_sources:
                if ref.chunk_id not in used_ids:
                    rec.evidence_chunk_ids.append(ref.chunk_id)
                    used_ids.add(ref.chunk_id)
                    resolved_sources.append(ref.source)
                    resolved_sources = sorted(set(resolved_sources))
            if len(resolved_sources) >= 3:
                break

    if len(rec.evidence_chunk_ids) < 3 and state.retrieved_chunks:
        used_ids = set(rec.evidence_chunk_ids)
        for ref in state.retrieved_chunks:
            if ref.chunk_id not in used_ids:
                rec.evidence_chunk_ids.append(ref.chunk_id)
                used_ids.add(ref.chunk_id)
                if ref.source and ref.source not in resolved_sources:
                    resolved_sources.append(ref.source)
                    resolved_sources = sorted(set(resolved_sources))
            if len(rec.evidence_chunk_ids) >= 5:
                break

    rec.evidence_sources = sorted(set(resolved_sources))
    return rec


def _parse_recommendations(text: str) -> list[dict]:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "recommendations" in data:
            recs = data["recommendations"]
            if isinstance(recs, list):
                return [r for r in recs if isinstance(r, dict)]
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
    except json.JSONDecodeError:
        pass
    return []


def decide_node(state: AgentState) -> dict[str, Any]:
    logger.info("Decider: synthesising recommendations")

    sources_by_chunk = _build_source_map(state)

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
    response = complete_json(prompt, temperature=0.3)
    raw_items = _parse_recommendations(response)

    recommendations: list[Recommendation] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            evidence_ids = raw.get("evidence_chunk_ids") or []
            if not isinstance(evidence_ids, list):
                evidence_ids = []
            rec = Recommendation(
                title=str(raw.get("title", "")).strip(),
                rationale=str(raw.get("rationale", "")).strip(),
                priority=str(raw.get("priority", "Medium")).strip().capitalize(),
                expected_impact=str(raw.get("expected_impact", "")).strip(),
                risk_assessment=str(raw.get("risk_assessment", "")).strip(),
                confidence=float(raw.get("confidence", 0.5)),
                evidence_chunk_ids=[str(c) for c in evidence_ids],
                evidence_sources=[],
            )
            rec = _enrich_evidence(rec, sources_by_chunk, state)
            if rec.title:
                recommendations.append(rec)
        except Exception as exc:
            logger.warning("Recommendation parse error: %s", exc)

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recommendations.sort(
        key=lambda r: (priority_order.get(r.priority, 1), -r.confidence)
    )

    logger.info("Decider: %d recommendations", len(recommendations))
    return {"recommendations": recommendations}
