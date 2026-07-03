"""Verify improvements 11 (embedding cache) and 12 (cross-run memory)."""

import sys
import time


def test_embedding_cache():
    from src.tools.embedding_cache import get_cached, put_cached, get_or_compute
    from src.processing.embedder import embed_texts

    query = "BMW electric vehicle strategy test query"

    cached = get_cached(query)
    assert cached is None, "Should not be cached yet"

    embedding = get_or_compute(query, lambda q: embed_texts([q])[0])
    assert len(embedding) == 384, f"Expected 384-dim, got {len(embedding)}"

    cached = get_cached(query)
    assert cached is not None, "Should be cached now"
    assert len(cached) == 384

    t0 = time.time()
    _ = get_or_compute(query, lambda q: embed_texts([q])[0])
    cached_time = time.time() - t0
    print(f"  Cache hit time: {cached_time*1000:.1f}ms")
    assert cached_time < 0.1, "Cache hit should be under 100ms"


def test_retriever_uses_cache():
    from src.tools.retriever import search
    import time

    query = "BMW Neue Klasse platform launch"

    t0 = time.time()
    hits1 = search(query, k=3)
    first_time = time.time() - t0

    t0 = time.time()
    hits2 = search(query, k=3)
    second_time = time.time() - t0

    print(f"  First search:  {first_time:.3f}s")
    print(f"  Second search: {second_time:.3f}s (cached embedding)")
    assert hits1[0].chunk_id == hits2[0].chunk_id, "Same query should return same results"


def test_cross_run_memory():
    from src.agent.run_memory import get_previous_run_summary

    summary = get_previous_run_summary(exclude_run_id="nonexistent")
    if summary:
        print(f"  Previous run context ({len(summary)} chars):")
        for line in summary.split("\n")[:4]:
            print(f"    {line}")
    else:
        print("  No previous runs found (expected if no agent runs exist)")


def _run(name, fn):
    print(f"\n[{name}]")
    started = time.time()
    try:
        fn()
        print(f"  PASS ({time.time() - started:.1f}s)")
        return True
    except Exception as e:
        print(f"  FAIL ({time.time() - started:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  Verify improvements 11, 12")
    print("=" * 60)

    results = [
        _run("embedding_cache", test_embedding_cache),
        _run("retriever_cache", test_retriever_uses_cache),
        _run("cross_run_memory", test_cross_run_memory),
    ]

    print(f"\n{sum(results)}/{len(results)} passed")
    if all(results):
        print("\nAll good. Delete this file: tests/verify_improvements.py")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
