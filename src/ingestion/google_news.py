"""Google News RSS scraper."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import quote_plus

import feedparser
from bs4 import BeautifulSoup

from src.config import GOOGLE_NEWS_RSS_URL, SEARCH_QUERIES
from src.ingestion.base import BaseScraper, Document, make_document_id


logger = logging.getLogger(__name__)

# News headlines are short by design; enforce a lower floor than press releases
_MIN_NEWS_LENGTH_CHARS = 60


class GoogleNewsScraper(BaseScraper):
    """Aggregate news via Google News RSS for the configured queries."""

    name = "google_news"

    def __init__(
        self,
        queries: list[str] | None = None,
        max_per_query: int = 50,
    ) -> None:
        self.queries = queries or SEARCH_QUERIES
        self.max_per_query = max_per_query

    def fetch(self) -> list[Document]:
        documents: dict[str, Document] = {}
        for query in self.queries:
            feed_url = GOOGLE_NEWS_RSS_URL.format(query=quote_plus(query))
            logger.info("Google News: querying %r", query)
            parsed = feedparser.parse(feed_url)

            if parsed.bozo:
                logger.warning(
                    "Feed parser warning for %r: %s",
                    query,
                    parsed.bozo_exception,
                )

            for entry in parsed.entries[: self.max_per_query]:
                doc = self._entry_to_document(entry, query)
                if doc is not None and doc.id not in documents:
                    documents[doc.id] = doc

        logger.info("Google News: %d unique documents", len(documents))
        return list(documents.values())

    @staticmethod
    def _entry_to_document(entry, query: str) -> Document | None:
        url = entry.get("link", "")
        title = entry.get("title", "").strip()
        if not url or not title:
            return None

        summary_html = entry.get("summary", "")
        summary_text = BeautifulSoup(summary_html, "lxml").get_text(
            separator=" ", strip=True
        )
        body = f"{title}\n\n{summary_text}".strip()
        if len(body) < _MIN_NEWS_LENGTH_CHARS:
            return None

        published_at: datetime | None = None
        if entry.get("published_parsed"):
            try:
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                published_at = None

        publisher = ""
        source_field = entry.get("source")
        if isinstance(source_field, dict):
            publisher = source_field.get("title", "")

        return Document(
            id=make_document_id(GoogleNewsScraper.name, url),
            source=GoogleNewsScraper.name,
            url=url,
            title=title,
            text=body,
            published_at=published_at,
            metadata={"query": query, "publisher": publisher},
        )
