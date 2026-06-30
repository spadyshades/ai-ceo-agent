"""Command-line entry point for the agent."""

from __future__ import annotations

import argparse
import time

from src.agent.graph import run_agent
from src.agent.schema import AgentState
from src.utils.logging import setup_logger


logger = setup_logger(__name__)


_DEFAULT_GOAL = (
    "What are the top strategic actions BMW should prioritise this quarter, "
    "based on the latest opportunities, risks, and trends?"
)


def _print_state(state: AgentState) -> None:
    print()
    print("=" * 72)
    print(f"  Agent run complete (run_id: {state.run_id})")
    print("=" * 72)

    print(f"\nGoal: {state.goal}\n")

    if state.plan:
        print(f"Plan ({len(state.plan.steps)} steps, {state.replan_count} replans):")
        for step in state.plan.steps:
            print(f"  {step.id}. {step.tool}  -- {step.description}")
        print()

    print(f"Tool calls executed: {len(state.tool_results)}")
    print(f"Opportunities found: {len(state.opportunities)}")
    print(f"Risks found:         {len(state.risks)}")
    print(f"Trends found:        {len(state.trends)}")

    if state.validation:
        verdict = "PASSED" if state.validation.passed else "FAILED"
        print(f"\nValidation: {verdict}")
        for issue in state.validation.issues:
            print(f"  - {issue}")

    if state.analysis:
        print(f"\nAnalysis summary:\n  {state.analysis.summary}")
        if state.analysis.key_findings:
            print("\nKey findings:")
            for f in state.analysis.key_findings:
                print(f"  - {f}")

    print(f"\nRecommendations ({len(state.recommendations)}):")
    for i, rec in enumerate(state.recommendations, 1):
        print(f"\n  {i}. [{rec.priority}] {rec.title}")
        print(f"     Confidence: {rec.confidence:.2f}")
        print(f"     Rationale: {rec.rationale}")
        print(f"     Expected impact: {rec.expected_impact}")
        print(f"     Risk assessment: {rec.risk_assessment}")
        print(
            f"     Evidence: {len(rec.evidence_chunk_ids)} chunks "
            f"from {len(set(rec.evidence_sources))} source(s)"
        )

    print("\nCEO briefing:")
    print(state.briefing or "(no briefing produced)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI CEO Agent")
    parser.add_argument(
        "goal",
        nargs="?",
        default=_DEFAULT_GOAL,
        help="Strategic question to ask the agent",
    )
    args = parser.parse_args()

    print(f"Goal: {args.goal!r}")
    print("Typical runtime on CPU: 4 to 10 minutes.\n")

    started = time.time()
    state = run_agent(args.goal)
    elapsed = time.time() - started

    _print_state(state)
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
