"""End-to-end smoke test for the Phase 4 agent.

Runs the full agent loop and verifies the result shape and SQLite persistence.
Total runtime is 5 to 10 minutes on CPU because the agent makes 4 LLM
synthesis calls plus internal LLM calls inside the intelligence engines.
"""

from __future__ import annotations

import sys
import time
import traceback

from src.utils.logging import setup_logger


logger = setup_logger(__name__)


def test_tools_registry() -> None:
    from src.agent.tools_registry import format_catalog, list_tools

    tools = list_tools()
    assert len(tools) >= 8, f"Expected at least 8 tools, got {len(tools)}: {tools}"
    catalog = format_catalog()
    for required in ("retriever", "detect_opportunities", "detect_risks", "detect_trends"):
        assert required in catalog, f"{required} missing from catalog"
    print(f"  Registered tools ({len(tools)}): {tools}")


def test_full_agent_run() -> None:
    from src.agent.graph import run_agent

    goal = "Give a short strategic snapshot of BMW based on the current corpus."
    state = run_agent(goal)

    assert state.plan is not None, "No plan was produced"
    assert state.plan.steps, "Plan has no steps"
    assert state.tool_results, "No tool calls executed"
    assert state.analysis is not None, "No analysis produced"
    assert state.recommendations, "No recommendations produced"
    assert state.validation is not None, "No validation performed"
    assert state.briefing, "No briefing produced"

    print(f"  Run ID:           {state.run_id}")
    print(f"  Plan steps:       {len(state.plan.steps)}")
    print(f"  Replan count:     {state.replan_count}")
    print(f"  Tool calls:       {len(state.tool_results)}")
    print(f"  Opportunities:    {len(state.opportunities)}")
    print(f"  Risks:            {len(state.risks)}")
    print(f"  Trends:           {len(state.trends)}")
    print(f"  Recommendations:  {len(state.recommendations)}")
    print(f"  Validation:       {'PASSED' if state.validation.passed else 'FAILED'}")
    print(f"  Briefing length:  {len(state.briefing)} chars")


def test_memory_persisted() -> None:
    from src.agent.memory import get_latest_run, list_runs

    latest = get_latest_run()
    assert latest is not None, "No agent runs persisted in SQLite"
    assert latest["status"] in ("success", "failed", "running"), \
        f"Unexpected status: {latest['status']}"
    runs = list_runs(limit=5)
    assert runs, "list_runs returned empty"
    print(f"  Runs in history: {len(runs)}; latest status: {latest['status']}")


def _run(name, fn):
    print(f"\n[{name}]")
    started = time.time()
    try:
        fn()
        elapsed = time.time() - started
        print(f"  PASS ({elapsed:.1f}s)")
        return True
    except Exception:
        elapsed = time.time() - started
        print(f"  FAIL ({elapsed:.1f}s)")
        traceback.print_exc()
        return False


def main():
    print("=" * 72)
    print("  Phase 4 smoke test (end-to-end agent run)")
    print("=" * 72)
    print("\nThis runs the full agent and typically takes 5 to 10 minutes.")

    tests = [
        ("tools_registry", test_tools_registry),
        ("full_agent_run", test_full_agent_run),
        ("memory_persisted", test_memory_persisted),
    ]
    results = [_run(name, fn) for name, fn in tests]

    print("\n" + "=" * 72)
    print(f"  {sum(results)}/{len(results)} tests passed")
    print("=" * 72)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
