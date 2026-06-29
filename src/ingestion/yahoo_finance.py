"""Yahoo Finance news scraper using the yfinance library."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import yfinance as yf

from src.config import COMPANY_TICKER
from src.ingestion.base import BaseScraper, Document, make_document_id


logger = logging.getLogger(__name__)

_MIN_NEWS_LENGTH_CHARS = 60


class YahooFinanceScraper(BaseScraper):
    """Fetch news headlines for the configured ticker via Yahoo Finance."""

    name = "yahoo_finance"

    def __init__(self, ticker: str = COMPANY_TICKER) -> None:
        self.ticker = ticker

    def fetch(self) -> list[Document]:
        try:
            yf_ticker = yf.Ticker(self.ticker)
            news_items = yf_ticker.news or []
        except Exception as exc:
            logger.error("Yahoo Finance fetch failed: %s", exc)
            return []

        documents: list[Document] = []
        for item in news_items:
            # yfinance has changed its news shape across versions; tolerate both.
            content = item.get("content", item) if isinstance(item, dict) else {}
            url = self._extract_url(content)
            title = content.get("title") or item.get("title", "")
            summary = (
                content.get("summary")
                or content.get("description")
                or item.get("summary", "")
            )

            if not url or not title:
                continue

            body = f"{title}\n\n{summary}".strip()
            if len(body) < _MIN_NEWS_LENGTH_CHARS:
                continue

            published_at = self._parse_published(
                content.get("pubDate") or item.get("providerPublishTime")
            )

            documents.append(
                Document(
                    id=make_document_id(self.name, url),
                    source=self.name,
                    url=url,
                    title=title,
                    text=body,
                    published_at=published_at,
                    metadata={
                        "ticker": self.ticker,
                        "provider": (content.get("provider") or {}).get(
                            "displayName", ""
                        ),
                    },
                )
            )

        logger.info("Yahoo Finance: %d documents", len(documents))
        return documents

    @staticmethod
    def _extract_url(content: dict) -> str:
        canonical = content.get("canonicalUrl")
        if isinstance(canonical, dict) and canonical.get("url"):
            return canonical["url"]
        return content.get("link") or content.get("clickThroughUrl", {}).get("url", "")

    @staticmethod
    def _parse_published(value) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return None
