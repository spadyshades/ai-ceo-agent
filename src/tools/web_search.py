"""External live news search via Google News RSS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote_plus

import feedparser
from bs4 import BeautifulSoup

from src.config import GOOGLE_NEWS_RSS_URL


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    published_at: datetime | None
    publisher: str


def search(query: str, limit: int = 10) -> list[SearchResult]:
    """Query Google News RSS for an arbitrary topic.

    This is the agent's window to information outside the indexed corpus.
    """
    feed_url = GOOGLE_NEWS_RSS_URL.format(query=quote_plus(query))
    parsed = feedparser.parse(feed_url)

    results: list[SearchResult] = []
    for entry in parsed.entries[:limit]:
        title = entry.get("title", "").strip()
        url = entry.get("link", "")
        if not title or not url:
            continue

        summary_html = entry.get("summary", "")
        snippet = BeautifulSoup(summary_html, "lxml").get_text(
            separator=" ", strip=True
        )

        published_at: datetime | None = None
        if entry.get("published_parsed"):
            try:
                published_at = datetime(
                    *entry.published_parsed[:6], tzinfo=timezone.utc
                )
            except (TypeError, ValueError):
                published_at = None

        publisher = ""
        source_field = entry.get("source")
        if isinstance(source_field, dict):
            publisher = source_field.get("title", "")

        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                published_at=published_at,
                publisher=publisher,
            )
        )
    return results
