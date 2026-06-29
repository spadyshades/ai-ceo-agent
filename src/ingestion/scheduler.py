"""Orchestrate all ingestion sources and persist results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.ingestion.arxiv_scraper import ArxivScraper
from src.ingestion.base import BaseScraper
from src.ingestion.bmw_press import BMWPressScraper
from src.ingestion.google_news import GoogleNewsScraper
from src.ingestion.hackernews import HackerNewsScraper
from src.ingestion.storage import (
    get_corpus_stats,
    log_ingestion_run,
    save_documents,
)
from src.ingestion.yahoo_finance import YahooFinanceScraper
from src.utils.logging import setup_logger


logger = setup_logger(__name__)


SCRAPERS: dict[str, type[BaseScraper]] = {
    "bmw_press": BMWPressScraper,
    "google_news": GoogleNewsScraper,
    "hackernews": HackerNewsScraper,
    "arxiv": ArxivScraper,
    "yahoo_finance": YahooFinanceScraper,
}


def run_scraper(name: str) -> int:
    """Run a single scraper, persist its output, and log the run."""
    scraper_class = SCRAPERS[name]
    scraper = scraper_class()
    started_at = datetime.now(timezone.utc)
    status = "success"
    error_message: str | None = None
    new_count = 0
    try:
        documents = scraper.fetch()
        new_count = save_documents(documents)
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
        logger.exception("Scraper %r failed", name)
    finally:
        finished_at = datetime.now(timezone.utc)
        log_ingestion_run(
            source=name,
            started_at=started_at,
            finished_at=finished_at,
            documents_fetched=new_count,
            status=status,
            error=error_message,
        )
    logger.info("Scraper %r: %d new documents (%s)", name, new_count, status)
    return new_count


def run_all() -> dict[str, int]:
    """Run every registered scraper."""
    return {name: run_scraper(name) for name in SCRAPERS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ingestion scrapers")
    parser.add_argument(
        "--source",
        choices=list(SCRAPERS.keys()) + ["all"],
        default="all",
        help="Which scraper to run (default: all)",
    )
    args = parser.parse_args()

    if args.source == "all":
        results = run_all()
    else:
        results = {args.source: run_scraper(args.source)}

    stats = get_corpus_stats()

    print("\nIngestion summary")
    print("-" * 60)
    for name, count in results.items():
        print(f"  {name:<20} {count:>4} new documents")
    print("-" * 60)
    print(f"  {'corpus total':<20} {stats['total_documents']:>4}")
    print(f"  {'distinct sources':<20} {stats['source_count']:>4}")
    print(f"  {'last run':<20} {stats['last_ingestion_at']}")


if __name__ == "__main__":
    main()
