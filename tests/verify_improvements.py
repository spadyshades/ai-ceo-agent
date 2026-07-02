"""Verify improvements 7 (competitor comparison), 8 (topic trends), 9 (query dedup), 10 (PDF report)."""

import sys
import time


def test_competitor_comparison():
    from src.tools.competitor_comparison import compare

    profiles = compare(competitors=["Tesla", "BYD", "Mercedes-Benz"])
    assert profiles, "No competitor profiles returned"
    for p in profiles:
        print(
            f"  {p.name}: {p.mention_count} mentions, "
            f"sentiment +{p.sentiment_positive}/~{p.sentiment_neutral}/-{p.sentiment_negative}"
        )


def test_topic_trends():
    from src.tools.topic_trends import detect_rising_topics

    topics = detect_rising_topics(days_recent=14, days_baseline=60, top_n=10)
    print(f"  {len(topics)} rising topics found")
    for t in topics[:5]:
        growth = "inf" if t.growth_rate == float("inf") else f"{t.growth_rate:.1f}x"
        print(f"    '{t.term}': recent={t.recent_tfidf:.4f}, growth={growth}")
    assert isinstance(topics, list)


def test_query_dedup():
    from src.tools.retriever import search
    from src.tools.retrieval_utils import dedupe_by_document

    hits = search("BMW electric vehicle strategy", k=10)
    deduped = dedupe_by_document(hits)
    print(f"  Before dedup: {len(hits)} hits")
    print(f"  After dedup:  {len(deduped)} hits (unique documents)")
    assert len(deduped) <= len(hits)


def test_pdf_report():
    from src.tools.report_generator import generate_report

    path = generate_report(output_path="data/test_report.pdf")
    print(f"  Report saved: {path}")

    import os
    size = os.path.getsize(path)
    print(f"  File size: {size:,} bytes")
    assert size > 500, "PDF too small, likely empty"
    os.remove(path)
    print("  Cleaned up test file")


def test_tools_in_registry():
    from src.agent.tools_registry import list_tools

    tools = list_tools()
    assert "compare_competitors" in tools, f"compare_competitors missing: {tools}"
    assert "detect_topic_trends" in tools, f"detect_topic_trends missing: {tools}"
    print(f"  Registered tools ({len(tools)}): {tools}")


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
    print("  Verify improvements 7, 8, 9, 10")
    print("=" * 60)

    results = [
        _run("competitor_comparison", test_competitor_comparison),
        _run("topic_trends", test_topic_trends),
        _run("query_dedup", test_query_dedup),
        _run("pdf_report", test_pdf_report),
        _run("tools_in_registry", test_tools_in_registry),
    ]

    print(f"\n{sum(results)}/{len(results)} passed")
    if all(results):
        print("\nAll good. Delete this file: tests/verify_improvements.py")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
