"""Planner node: produces a structured tool-execution plan."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.schema import AgentState, Plan, PlanStep
from src.agent.tools_registry import format_catalog, list_tools
from src.agent.run_memory import get_previous_run_summary
from src.utils.llm import complete_json


logger = logging.getLogger(__name__)


_PLAN_PROMPT = """You are the planning component of a strategic intelligence agent for the CEO of BMW.

User goal: {goal}

{previous_context}
{replan_context}
Available tools:
{tool_catalog}

Produce a plan with 3 to 7 steps to achieve the goal. Each step calls exactly one tool.

Guidelines:
- For broad strategic goals, include detect_opportunities, detect_risks, and detect_trends.
- For specific questions, prefer targeted retriever queries (much faster than engines).
- Use web_search for information outside the BMW corpus.
- Use hybrid_search when the query contains specific model names or technical terms.
- Use financial_data to include live stock market context.
- Use compare_competitors when the goal involves competitive positioning.
- Use detect_topic_trends to find rising themes beyond named entities.
- If previous analysis exists, build on it rather than repeating the same queries.
- Keep the plan minimal; each LLM-based tool takes about 60 seconds on CPU.

Return a JSON object with this exact shape:
{{
  "reasoning": "brief explanation of the plan strategy",
  "steps": [
    {{"id": 1, "tool": "tool_name", "params": {{}}, "description": "what this step accomplishes"}}
  ]
}}

JSON:"""


_REPLAN_PREFIX = """The previous plan failed validation with these issues:
{issues}

Please produce a revised plan that addresses them. Gather more diverse
evidence, use more sources, or refine the targeting.

"""

_PREVIOUS_CONTEXT_PREFIX = """Context from the most recent previous analysis:
{summary}

Build on this prior analysis where relevant. Avoid repeating identical queries.

"""


def _fallback_plan(goal: str) -> Plan:
    return Plan(
        reasoning="fallback default plan: broad strategic scan with targeted retrieval",
        steps=[
            PlanStep(id=1, tool="detect_opportunities", params={},
                     description="Identify strategic opportunities"),
            PlanStep(id=2, tool="detect_risks", params={},
                     description="Identify strategic risks"),
            PlanStep(id=3, tool="detect_trends", params={},
                     description="Identify emerging trends"),
            PlanStep(id=4, tool="retriever", params={"query": goal, "k": 5},
                     description="Retrieve evidence directly tied to the goal"),
        ],
    )


def planner_node(state: AgentState) -> dict[str, Any]:
    logger.info("Planner: replan_count=%d", state.replan_count)

    replan_context = ""
    if state.validation and not state.validation.passed:
        issues = "\n".join(f"- {iss}" for iss in state.validation.issues)
        replan_context = _REPLAN_PREFIX.format(issues=issues)

    previous_context = ""
    if state.replan_count == 0:
        prev_summary = get_previous_run_summary(exclude_run_id=state.run_id)
        if prev_summary:
            previous_context = _PREVIOUS_CONTEXT_PREFIX.format(summary=prev_summary)

    prompt = _PLAN_PROMPT.format(
        goal=state.goal,
        previous_context=previous_context,
        replan_context=replan_context,
        tool_catalog=format_catalog(),
    )
    response = complete_json(prompt, temperature=0.2)

    plan: Plan | None = None
    try:
        parsed = json.loads(response)
        plan = Plan.model_validate(parsed)
    except Exception as exc:
        logger.warning("Plan parse failed: %s", exc)

    if plan is None:
        logger.warning("Falling back to default plan")
        plan = _fallback_plan(state.goal)

    known_tools = set(list_tools())
    plan.steps = [s for s in plan.steps if s.tool in known_tools]
    if not plan.steps:
        plan = _fallback_plan(state.goal)

    logger.info("Planner: produced %d-step plan", len(plan.steps))
    return {
        "plan": plan,
        "replan_count": state.replan_count + 1,
        "next_step_index": 0,
    }
