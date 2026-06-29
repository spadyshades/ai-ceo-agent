"""BMW Group press release scraper."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.config import BMW_PRESS_LIST_URL, MIN_DOC_LENGTH_CHARS, USER_AGENT
from src.ingestion.base import BaseScraper, Document, make_document_id


logger = logging.getLogger(__name__)


class BMWPressScraper(BaseScraper):
    """Scrape press releases from press.bmwgroup.com."""

    name = "bmw_press"

    _ARTICLE_LINK_PATTERN = re.compile(r"/global/article/detail/")

    def __init__(
        self,
        list_url: str = BMW_PRESS_LIST_URL,
        max_articles: int = 50,
        request_delay_seconds: float = 1.0,
        timeout_seconds: int = 20,
    ) -> None:
        self.list_url = list_url
        self.max_articles = max_articles
        self.request_delay_seconds = request_delay_seconds
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch(self) -> list[Document]:
        try:
            article_urls = self._fetch_listing()
        except requests.RequestException as exc:
            logger.error("Failed to fetch BMW press listing: %s", exc)
            return []

        logger.info("BMW press: found %d article URLs", len(article_urls))

        documents: list[Document] = []
        for url in article_urls[: self.max_articles]:
            try:
                doc = self._fetch_article(url)
            except requests.RequestException as exc:
                logger.warning("Skipping %s: %s", url, exc)
                continue
            if doc is not None:
                documents.append(doc)
            time.sleep(self.request_delay_seconds)

        logger.info("BMW press: extracted %d documents", len(documents))
        return documents

    def _fetch_listing(self) -> list[str]:
        response = self.session.get(self.list_url, timeout=self.timeout_seconds)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        urls: list[str] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=self._ARTICLE_LINK_PATTERN):
            href = anchor.get("href")
            if not href:
                continue
            full_url = urljoin(self.list_url, href)
            if full_url not in seen:
                urls.append(full_url)
                seen.add(full_url)
        return urls

    def _fetch_article(self, url: str) -> Document | None:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        title = self._extract_title(soup)
        body = self._extract_body(soup)
        published_at = self._extract_date(soup)

        if not title or len(body) < MIN_DOC_LENGTH_CHARS:
            logger.debug("Skipping short or untitled article: %s", url)
            return None

        return Document(
            id=make_document_id(self.name, url),
            source=self.name,
            url=url,
            title=title,
            text=body,
            published_at=published_at,
            metadata={"company": "BMW", "type": "press_release"},
        )

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        for selector in ("h1", "meta[property='og:title']", "title"):
            element = soup.select_one(selector)
            if not element:
                continue
            if element.name == "meta":
                return element.get("content", "").strip()
            return element.get_text(strip=True)
        return ""

    @staticmethod
    def _extract_body(soup: BeautifulSoup) -> str:
        candidates = (
            "div.press-release-content",
            "div.article-body",
            "div[itemprop='articleBody']",
            "main",
            "article",
        )
        for selector in candidates:
            element = soup.select_one(selector)
            if not element:
                continue
            for tag in element(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            text = element.get_text(separator="\n", strip=True)
            if len(text) >= MIN_DOC_LENGTH_CHARS:
                return text

        # Fallback when no semantic container matches
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return "\n\n".join(p for p in paragraphs if p)

    @staticmethod
    def _extract_date(soup: BeautifulSoup) -> datetime | None:
        for selector in (
            "meta[property='article:published_time']",
            "meta[name='date']",
            "time[datetime]",
        ):
            element = soup.select_one(selector)
            if not element:
                continue
            raw = element.get("content") or element.get("datetime")
            if not raw:
                continue
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
        return None
