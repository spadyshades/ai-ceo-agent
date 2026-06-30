"""Planner node: produces a structured tool-execution plan."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.agent.schema import AgentState, Plan, PlanStep
from src.agent.tools_registry import format_catalog, list_tools
from src.utils.llm import complete


logger = logging.getLogger(__name__)


_PLAN_PROMPT = """You are the planning component of a strategic intelligence agent for the CEO of BMW.

User goal: {goal}

{replan_context}
Available tools:
{tool_catalog}

Produce a plan with 3 to 7 steps to achieve the goal. Each step calls exactly one tool.

Guidelines:
- For broad strategic goals, include detect_opportunities, detect_risks, and detect_trends.
- For specific questions, prefer targeted retriever queries (much faster than engines).
- Use web_search for information outside the BMW corpus.
- Keep the plan minimal; each LLM-based tool takes about 60 seconds on CPU.

Return ONLY a valid JSON object with this exact shape:
{{
  "reasoning": "brief explanation of the plan strategy",
  "steps": [
    {{"id": 1, "tool": "tool_name", "params": {{}}, "description": "what this step accomplishes"}}
  ]
}}

No commentary before or after the JSON. Do not wrap in markdown.

JSON:"""


_REPLAN_PREFIX = """The previous plan failed validation with these issues:
{issues}

Please produce a revised plan that addresses them. Gather more diverse
evidence, use more sources, or refine the targeting.

"""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse plan JSON: %s", exc)
        return None


def _fallback_plan(goal: str) -> Plan:
    """Default plan used when the LLM cannot produce a valid one."""
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
    """Produce a plan for the current goal."""
    logger.info("Planner: replan_count=%d", state.replan_count)

    replan_context = ""
    if state.validation and not state.validation.passed:
        issues = "\n".join(f"- {iss}" for iss in state.validation.issues)
        replan_context = _REPLAN_PREFIX.format(issues=issues)

    prompt = _PLAN_PROMPT.format(
        goal=state.goal,
        replan_context=replan_context,
        tool_catalog=format_catalog(),
    )
    response = complete(prompt, temperature=0.2)

    parsed = _extract_json_object(response)
    plan: Plan | None = None
    if parsed:
        try:
            plan = Plan.model_validate(parsed)
        except Exception as exc:
            logger.warning("Plan validation failed: %s", exc)

    if plan is None:
        logger.warning("Falling back to default plan")
        plan = _fallback_plan(state.goal)

    # Filter out steps referencing unknown tools
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
