"""Analyzer node: synthesise gathered evidence into a strategic analysis."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.agent.schema import AgentState, AnalysisOutput
from src.utils.llm import complete


logger = logging.getLogger(__name__)


_PROMPT = """You are the analysis component of a strategic intelligence agent for BMW.

User goal: {goal}

Evidence gathered:

Opportunities ({n_opp}):
{opp_summary}

Risks ({n_risks}):
{risk_summary}

Trends ({n_trends}):
{trend_summary}

Retrieved chunks available for citation:
{retrieved_summary}

Tool call summaries:
{tool_summary}

Synthesise this evidence into a coherent executive analysis.

Return ONLY a valid JSON object with this exact shape:
{{
  "summary": "3-5 sentence executive summary of the strategic situation",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "supporting_evidence_chunk_ids": ["chunk_id_1", "chunk_id_2"]
}}

Cite real chunk IDs from the retrieved chunks or evidence above.
No commentary before or after.

JSON:"""


def _format_items(items, kind: str) -> str:
    if not items:
        return f"(no {kind} identified)"
    return "\n".join(
        f"- [{it.impact}] {it.title}: {it.description[:200]}"
        for it in items[:8]
    )


def _format_retrieved(retrieved_chunks, limit: int = 30) -> str:
    if not retrieved_chunks:
        return "(none)"
    return "\n".join(
        f"- {ref.chunk_id} [{ref.source}] {ref.title[:90]}"
        for ref in retrieved_chunks[:limit]
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def analyze_node(state: AgentState) -> dict[str, Any]:
    logger.info("Analyzer: synthesising evidence")

    tool_summary = "\n".join(
        f"- {r.tool}: {r.summary}" + (f" [ERROR: {r.error}]" if r.error else "")
        for r in state.tool_results
    ) or "(no tool calls recorded)"

    prompt = _PROMPT.format(
        goal=state.goal,
        n_opp=len(state.opportunities),
        n_risks=len(state.risks),
        n_trends=len(state.trends),
        opp_summary=_format_items(state.opportunities, "opportunities"),
        risk_summary=_format_items(state.risks, "risks"),
        trend_summary=_format_items(state.trends, "trends"),
        retrieved_summary=_format_retrieved(state.retrieved_chunks),
        tool_summary=tool_summary,
    )
    response = complete(prompt, temperature=0.3)
    parsed = _extract_json_object(response)

    analysis: AnalysisOutput
    if parsed:
        try:
            analysis = AnalysisOutput.model_validate(parsed)
        except Exception as exc:
            logger.warning("Analysis validation failed: %s", exc)
            analysis = AnalysisOutput(summary="Analysis synthesis failed.")
    else:
        analysis = AnalysisOutput(
            summary="Analysis synthesis returned malformed output."
        )

    return {"analysis": analysis}
