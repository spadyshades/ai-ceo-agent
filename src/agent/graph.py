"""LangGraph state machine wiring all agent nodes together."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.analyzer import analyze_node
from src.agent.briefing import briefing_node
from src.agent.decider import decide_node
from src.agent.executor import execute_step_node, route_after_execute
from src.agent.memory import (
    finish_run,
    record_plan,
    record_recommendations,
    record_tool_call,
    record_validation,
    start_run,
)
from src.agent.planner import planner_node
from src.agent.schema import AgentState
from src.agent.validator import route_after_validate, validate_node


logger = logging.getLogger(__name__)


def _planner_with_persist(state: AgentState) -> dict[str, Any]:
    update = planner_node(state)
    plan = update.get("plan")
    if state.run_id and plan is not None:
        try:
            record_plan(state.run_id, plan, state.replan_count)
        except Exception as exc:
            logger.warning("Failed to persist plan: %s", exc)
    return update


def _executor_with_persist(state: AgentState) -> dict[str, Any]:
    update = execute_step_node(state)
    # Persist the most recently appended tool call
    if state.run_id and "tool_results" in update:
        new_calls = update["tool_results"][len(state.tool_results):]
        for call in new_calls:
            try:
                record_tool_call(state.run_id, call)
            except Exception as exc:
                logger.warning("Failed to persist tool call: %s", exc)
    return update


def _validate_with_persist(state: AgentState) -> dict[str, Any]:
    update = validate_node(state)
    validation = update.get("validation")
    if state.run_id and validation is not None:
        try:
            record_validation(state.run_id, validation)
        except Exception as exc:
            logger.warning("Failed to persist validation: %s", exc)
    return update


def build_graph():
    """Construct and compile the agent state graph."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", _planner_with_persist)
    graph.add_node("execute_step", _executor_with_persist)
    graph.add_node("analyze", analyze_node)
    graph.add_node("decide", decide_node)
    graph.add_node("validate", _validate_with_persist)
    graph.add_node("briefing", briefing_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "execute_step")
    graph.add_conditional_edges(
        "execute_step",
        route_after_execute,
        {"execute_step": "execute_step", "analyze": "analyze"},
    )
    graph.add_edge("analyze", "decide")
    graph.add_edge("decide", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"planner": "planner", "briefing": "briefing"},
    )
    graph.add_edge("briefing", END)

    return graph.compile()


def run_agent(goal: str) -> AgentState:
    """Run the agent against a goal and return the final state."""
    run_id = start_run(goal)
    logger.info("Agent run started: %s", run_id)

    initial = AgentState(goal=goal, run_id=run_id)
    compiled = build_graph()

    try:
        result = compiled.invoke(initial)
        # LangGraph may return either a dict or an AgentState; normalise.
        if isinstance(result, AgentState):
            final_state = result
        elif isinstance(result, dict):
            final_state = AgentState.model_validate(result)
        else:
            final_state = AgentState.model_validate(dict(result))

        if final_state.recommendations:
            try:
                record_recommendations(
                    run_id,
                    final_state.recommendations,
                    validated=bool(
                        final_state.validation and final_state.validation.passed
                    ),
                )
            except Exception as exc:
                logger.warning("Failed to persist recommendations: %s", exc)

        finish_run(run_id, "success", final_state)
        return final_state
    except Exception as exc:
        logger.exception("Agent run failed")
        partial = AgentState(goal=goal, run_id=run_id, errors=[str(exc)])
        finish_run(run_id, "failed", partial, error=str(exc))
        raise
