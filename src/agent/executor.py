"""Executor node: runs the next planned tool call."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.schema import (
    AgentState,
    RetrievedChunkRef,
    StrategicItem,
    ToolCallResult,
)
from src.agent.tools_registry import execute, summarize


logger = logging.getLogger(__name__)


def _to_schema_items(raw_items) -> list[StrategicItem]:
    converted: list[StrategicItem] = []
    for item in raw_items:
        converted.append(
            StrategicItem(
                title=getattr(item, "title", ""),
                description=getattr(item, "description", ""),
                impact=getattr(item, "impact", "Medium"),
                confidence=getattr(item, "confidence", 0.5),
                evidence_chunk_ids=list(getattr(item, "evidence_chunk_ids", []) or []),
                evidence_sources=list(getattr(item, "evidence_sources", []) or []),
            )
        )
    return converted


def _extract_chunk_refs(raw_result) -> list[RetrievedChunkRef]:
    """Pull chunk references out of a retriever result."""
    refs: list[RetrievedChunkRef] = []
    if not isinstance(raw_result, list):
        return refs
    for hit in raw_result:
        chunk_id = getattr(hit, "chunk_id", None)
        if not chunk_id:
            continue
        refs.append(
            RetrievedChunkRef(
                chunk_id=chunk_id,
                source=getattr(hit, "source", ""),
                title=(getattr(hit, "title", "") or "")[:160],
                snippet=(getattr(hit, "text", "") or "")[:200],
            )
        )
    return refs


def execute_step_node(state: AgentState) -> dict[str, Any]:
    if state.plan is None or state.next_step_index >= len(state.plan.steps):
        return {}

    step = state.plan.steps[state.next_step_index]
    logger.info(
        "Executor: step %d (%d/%d) tool=%s params=%s",
        step.id,
        state.next_step_index + 1,
        len(state.plan.steps),
        step.tool,
        step.params,
    )

    extra_updates: dict[str, Any] = {}
    summary = ""
    error_message = ""

    try:
        raw_result = execute(step.tool, step.params)
        summary = summarize(step.tool, raw_result)

        if step.tool == "detect_opportunities":
            extra_updates["opportunities"] = (
                state.opportunities + _to_schema_items(raw_result)
            )
        elif step.tool == "detect_risks":
            extra_updates["risks"] = state.risks + _to_schema_items(raw_result)
        elif step.tool == "detect_trends":
            _rising, synthesised = raw_result
            extra_updates["trends"] = state.trends + _to_schema_items(synthesised)
        elif step.tool == "retriever":
            new_refs = _extract_chunk_refs(raw_result)
            if new_refs:
                existing_ids = {r.chunk_id for r in state.retrieved_chunks}
                unique = [r for r in new_refs if r.chunk_id not in existing_ids]
                if unique:
                    extra_updates["retrieved_chunks"] = state.retrieved_chunks + unique
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        logger.exception("Tool execution failed: %s", step.tool)

    tool_result = ToolCallResult(
        step_id=step.id,
        tool=step.tool,
        params=step.params,
        summary=summary,
        error=error_message,
    )

    return {
        "tool_results": state.tool_results + [tool_result],
        "next_step_index": state.next_step_index + 1,
        **extra_updates,
    }


def route_after_execute(state: AgentState) -> str:
    if state.plan is None or state.next_step_index >= len(state.plan.steps):
        return "analyze"
    return "execute_step"
