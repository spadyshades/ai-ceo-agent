"""Briefing node: produce the 3-question CEO briefing."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.schema import AgentState
from src.utils.llm import complete


logger = logging.getLogger(__name__)


_PROMPT = """You are preparing the CEO briefing for BMW's executive team.

User goal: {goal}

Analysis summary: {analysis_summary}

Top recommendations:
{recs}

Write a concise executive briefing in plain prose, no markdown, no bullet points,
that answers, in order:
1. What happened? (the situation, 2-3 sentences)
2. Why does it matter? (the strategic significance, 2-3 sentences)
3. What should management do next? (top 2-3 actions, 2-3 sentences)

Keep the total under 200 words. Use natural connecting language between the three
parts; do not label them.

Briefing:"""


def briefing_node(state: AgentState) -> dict[str, Any]:
    """Produce the CEO briefing."""
    logger.info("Briefing: composing executive summary")

    analysis_summary = (
        state.analysis.summary if state.analysis
        else "No analysis was produced."
    )
    recs_text = (
        "\n".join(
            f"- [{r.priority}] {r.title}: {r.rationale[:200]}"
            for r in state.recommendations[:5]
        )
        if state.recommendations
        else "(no recommendations available)"
    )

    prompt = _PROMPT.format(
        goal=state.goal,
        analysis_summary=analysis_summary,
        recs=recs_text,
    )
    briefing = complete(prompt, temperature=0.3).strip()
    return {"briefing": briefing}
