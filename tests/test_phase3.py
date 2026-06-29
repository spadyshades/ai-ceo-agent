"""Smoke test for the Phase 3 tools and intelligence engines.

Run from project root: python -m tests.test_phase3

The script exercises each tool in turn. Failures are reported but do not
abort the run, so a partial failure (e.g. the LLM being slow) does not
prevent the other tools from being verified.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Callable

from src.utils.logging import setup_logger


logger = setup_logger(__name__)


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _run(name: str, fn: Callable[[], None]) -> bool:
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


def test_retriever() -> None:
    from src.tools.retriever import search, search_by_entity

    hits = search("BMW Neue Klasse platform", k=3)
    assert hits, "Retriever returned no results"
    for hit in hits:
        assert hit.chunk_id and hit.text and 0.0 <= hit.similarity <= 1.0
    print(f"  top hit: {hits[0].title[:80]} (sim={hits[0].similarity:.3f})")

    by_entity = search_by_entity("electric vehicle competition", "Mercedes-Benz", k=3)
    print(f"  by_entity returned {len(by_entity)} chunks mentioning Mercedes-Benz")


def test_web_search() -> None:
    from src.tools.web_search import search

    results = search("BMW iX3 Neue Klasse", limit=5)
    assert results, "Web search returned no results"
    print(f"  fetched {len(results)} external results")
    print(f"  example: {results[0].title[:80]}")


def test_sentiment() -> None:
    from src.tools.sentiment import classify, classify_batch

    pos = classify("BMW posted record-breaking sales and strong margins.")
    neg = classify("BMW faces a major recall and falling demand in China.")
    neu = classify("BMW announced the date for its next press conference.")
    print(f"  positive sample -> {pos.label} ({pos.score:.3f})")
    print(f"  negative sample -> {neg.label} ({neg.score:.3f})")
    print(f"  neutral sample  -> {neu.label} ({neu.score:.3f})")

    batch = classify_batch([
        "Strong quarter for BMW",
        "Severe supply chain problems for automakers",
    ])
    assert len(batch) == 2


def test_entity_extractor() -> None:
    from src.tools.entity_extractor import extract

    sample = (
        "BMW Group announced a new partnership with Qualcomm to expand "
        "autonomous driving capabilities in China and Germany."
    )
    entities = extract(sample)
    print(f"  labels found: {sorted(entities.keys())}")
    orgs = entities.get("ORG", [])
    assert orgs, "No organizations extracted"
    print(f"  ORG entities: {orgs}")


def test_trend_detector() -> None:
    from src.tools.trend_detector import detect_rising_entities

    trends = detect_rising_entities(label="ORG", days_recent=14, days_baseline=60, min_count=2)
    print(f"  found {len(trends)} trending organisations")
    for t in trends[:5]:
        growth = "inf" if t.growth_rate == float("inf") else f"{t.growth_rate:.2f}"
        print(f"    {t.entity}: recent={t.recent_count}, baseline={t.baseline_count}, growth={growth}")


def test_source_credibility() -> None:
    from src.tools.source_credibility import get_source_tier, score_source

    for source in ("bmw_press", "google_news", "hackernews", "unknown_source"):
        tier = get_source_tier(source)
        score = score_source(source, "2026-06-20T12:00:00+00:00")
        print(f"  {source:<15} tier={tier}, score={score}")


def test_comparator() -> None:
    from src.tools.comparator import (
        find_contradicting_evidence,
        find_supporting_evidence,
        negate_claim,
    )

    claim = "BMW is leading the European electric vehicle market"
    negation = negate_claim(claim)
    print(f"  claim:    {claim}")
    print(f"  negation: {negation}")

    support = find_supporting_evidence(claim, k=2)
    contradict = find_contradicting_evidence(claim, k=2)
    print(f"  supporting hits: {len(support)}; contradicting hits: {len(contradict)}")
    if support:
        print(f"    top support: {support[0].title[:80]}")
    if contradict:
        print(f"    top contradict: {contradict[0].title[:80]}")


def test_llm() -> None:
    from src.utils.llm import complete

    response = complete("Reply with exactly: 'tool layer reachable'.", temperature=0.0)
    print(f"  LLM reply: {response.strip()[:80]}")


def test_intelligence_engines() -> None:
    from src.intelligence.engines import (
        detect_opportunities,
        detect_risks,
        detect_trends,
        score_items_by_credibility,
    )

    print("  detecting opportunities ...")
    opportunities = score_items_by_credibility(detect_opportunities())
    print(f"  opportunities: {len(opportunities)}")
    for item in opportunities[:3]:
        print(f"    [{item.impact}] {item.title} (conf={item.confidence:.2f})")

    print("  detecting risks ...")
    risks = score_items_by_credibility(detect_risks())
    print(f"  risks: {len(risks)}")
    for item in risks[:3]:
        print(f"    [{item.impact}] {item.title} (conf={item.confidence:.2f})")

    print("  detecting trends ...")
    rising, trends = detect_trends(top_entities=8)
    trends = score_items_by_credibility(trends)
    print(f"  rising entities: {len(rising)}, synthesised trends: {len(trends)}")
    for item in trends[:3]:
        print(f"    [{item.impact}] {item.title} (conf={item.confidence:.2f})")


def main() -> None:
    _section("Phase 3 smoke test - atomic tools")
    fast_tests = [
        ("retriever", test_retriever),
        ("web_search", test_web_search),
        ("entity_extractor", test_entity_extractor),
        ("source_credibility", test_source_credibility),
        ("trend_detector", test_trend_detector),
        ("sentiment", test_sentiment),
        ("llm", test_llm),
    ]
    fast_results = [_run(name, fn) for name, fn in fast_tests]

    _section("Phase 3 smoke test - LLM-using tools (slower)")
    slow_tests = [
        ("comparator", test_comparator),
        ("intelligence_engines", test_intelligence_engines),
    ]
    slow_results = [_run(name, fn) for name, fn in slow_tests]

    _section("Summary")
    total = len(fast_results) + len(slow_results)
    passed = sum(fast_results) + sum(slow_results)
    print(f"\n  {passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
