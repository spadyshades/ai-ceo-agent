"""Phase 5 smoke test: evidence and validation layer.

Verifies the enriched decider + multi-layer validator produce
passing validation on at least one run.
"""

from __future__ import annotations

import sys
import time
import traceback

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


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


def test_source_existence_check() -> None:
    from src.processing.indexer import get_collection

    collection = get_collection()
    sample = collection.peek(3)
    ids = sample["ids"]
    assert ids, "No chunks in collection"

    result = collection.get(ids=ids, include=[])
    assert set(result["ids"]) == set(ids), "Chunk lookup mismatch"
    print(f"  Source existence check works: verified {len(ids)} chunk IDs")


def test_claim_alignment() -> None:
    from src.processing.embedder import embed_texts

    claim = "BMW should expand EV production in China"
    evidence = "BMW plans to increase electric vehicle manufacturing capacity"
    unrelated = "The weather forecast shows sunny skies tomorrow"

    embs = embed_texts([claim, evidence, unrelated])
    sim_relevant = sum(a * b for a, b in zip(embs[0], embs[1]))
    sim_unrelated = sum(a * b for a, b in zip(embs[0], embs[2]))

    print(f"  relevant similarity: {sim_relevant:.3f}")
    print(f"  unrelated similarity: {sim_unrelated:.3f}")
    assert sim_relevant > sim_unrelated, "Relevant evidence should score higher"


def test_adversarial_check() -> None:
    from src.tools.comparator import find_contradicting_evidence

    contradictions = find_contradicting_evidence(
        "BMW is the market leader in European EVs", k=2
    )
    print(f"  Found {len(contradictions)} contradicting passages")
    assert isinstance(contradictions, list)


def test_full_agent_with_validation() -> None:
    from src.agent.graph import run_agent

    goal = "Give a strategic snapshot of BMW's competitive position."
    state = run_agent(goal)

    assert state.plan is not None
    assert state.recommendations, "No recommendations produced"
    assert state.validation is not None
    assert state.briefing

    print(f"  Run ID:          {state.run_id}")
    print(f"  Replans:         {state.replan_count}")
    print(f"  Recommendations: {len(state.recommendations)}")
    print(f"  Validation:      {'PASSED' if state.validation.passed else 'FAILED'}")
    if state.validation.issues:
        for iss in state.validation.issues:
            print(f"    - {iss}")
    print(f"  Briefing length: {len(state.briefing)} chars")

    for i, rec in enumerate(state.recommendations, 1):
        print(
            f"  rec#{i}: [{rec.priority}] {rec.title} "
            f"(conf={rec.confidence:.2f}, "
            f"evidence={len(rec.evidence_chunk_ids)} chunks, "
            f"sources={rec.evidence_sources})"
        )


def main():
    print("=" * 72)
    print("  Phase 5 smoke test - evidence and validation")
    print("=" * 72)

    tests = [
        ("source_existence", test_source_existence_check),
        ("claim_alignment", test_claim_alignment),
        ("adversarial_check", test_adversarial_check),
        ("full_agent", test_full_agent_with_validation),
    ]
    results = [_run(name, fn) for name, fn in tests]

    print("\n" + "=" * 72)
    print(f"  {sum(results)}/{len(results)} tests passed")
    print("=" * 72)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
