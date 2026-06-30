"""Validator node (Phase 4: rule-based; Phase 5 adds adversarial LLM check)."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.schema import AgentState, ValidationResult
from src.config import (
    MAX_REPLAN_ATTEMPTS,
    MIN_DISTINCT_SOURCES_PER_RECOMMENDATION,
    MIN_EVIDENCE_PER_RECOMMENDATION,
)


logger = logging.getLogger(__name__)


def validate_node(state: AgentState) -> dict[str, Any]:
    """Apply rule-based checks to the agent's recommendations."""
    logger.info("Validator: checking %d recommendations", len(state.recommendations))

    issues: list[str] = []

    if not state.recommendations:
        issues.append("No recommendations were produced")
    else:
        for i, rec in enumerate(state.recommendations, start=1):
            label = f"recommendation #{i} ('{rec.title[:40]}')"

            if len(rec.evidence_chunk_ids) < MIN_EVIDENCE_PER_RECOMMENDATION:
                issues.append(
                    f"{label} has {len(rec.evidence_chunk_ids)} evidence chunks "
                    f"(need at least {MIN_EVIDENCE_PER_RECOMMENDATION})"
                )

            distinct_sources = len(set(rec.evidence_sources))
            if distinct_sources < MIN_DISTINCT_SOURCES_PER_RECOMMENDATION:
                issues.append(
                    f"{label} draws from {distinct_sources} distinct source(s) "
                    f"(need at least {MIN_DISTINCT_SOURCES_PER_RECOMMENDATION})"
                )

            if rec.confidence < 0.2:
                issues.append(
                    f"{label} has very low confidence ({rec.confidence:.2f})"
                )

            if not rec.rationale.strip():
                issues.append(f"{label} is missing rationale")

    validation = ValidationResult(passed=len(issues) == 0, issues=issues)
    logger.info(
        "Validator: %s with %d issue(s)",
        "PASSED" if validation.passed else "FAILED",
        len(issues),
    )
    return {"validation": validation}


def route_after_validate(state: AgentState) -> str:
    """Decide whether to finalise or re-plan."""
    if state.validation and state.validation.passed:
        return "briefing"
    if state.replan_count >= MAX_REPLAN_ATTEMPTS:
        logger.info(
            "Validator: replan limit reached (%d); proceeding to briefing",
            state.replan_count,
        )
        return "briefing"
    logger.info("Validator: re-planning")
    return "planner"
