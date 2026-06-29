"""Hacker News scraper via the Algolia search API."""

from __future__ import annotations

import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from src.config import HN_ALGOLIA_API, SEARCH_QUERIES, USER_AGENT
from src.ingestion.base import BaseScraper, Document, make_document_id


logger = logging.getLogger(__name__)

_MIN_HN_LENGTH_CHARS = 60


class HackerNewsScraper(BaseScraper):
    """Fetch stories and comments from Hacker News for the configured queries."""

    name = "hackernews"

    def __init__(
        self,
        queries: list[str] | None = None,
        hits_per_page: int = 50,
        timeout_seconds: int = 20,
    ) -> None:
        self.queries = queries or SEARCH_QUERIES
        self.hits_per_page = hits_per_page
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch(self) -> list[Document]:
        documents: dict[str, Document] = {}
        for query in self.queries:
            for tag in ("story", "comment"):
                hits = self._search(query=query, tag=tag)
                for hit in hits:
                    doc = self._hit_to_document(hit, query)
                    if doc is not None and doc.id not in documents:
                        documents[doc.id] = doc

        logger.info("Hacker News: %d unique documents", len(documents))
        return list(documents.values())

    def _search(self, query: str, tag: str) -> list[dict]:
        params = {
            "query": query,
            "tags": tag,
            "hitsPerPage": self.hits_per_page,
        }
        try:
            response = self.session.get(
                HN_ALGOLIA_API, params=params, timeout=self.timeout_seconds
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "HN request failed for query=%r tag=%r: %s", query, tag, exc
            )
            return []
        return response.json().get("hits", [])

    @classmethod
    def _hit_to_document(cls, hit: dict, query: str) -> Document | None:
        object_id = hit.get("objectID")
        if not object_id:
            return None

        url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        title = hit.get("title") or hit.get("story_title") or ""
        raw_text = hit.get("story_text") or hit.get("comment_text") or ""
        text = BeautifulSoup(raw_text, "lxml").get_text(separator=" ", strip=True)

        body = f"{title}\n\n{text}".strip() if title else text
        if len(body) < _MIN_HN_LENGTH_CHARS:
            return None

        published_at: datetime | None = None
        created_at = hit.get("created_at")
        if created_at:
            try:
                published_at = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                )
            except ValueError:
                published_at = None

        return Document(
            id=make_document_id(cls.name, url),
            source=cls.name,
            url=url,
            title=title or f"HN item {object_id}",
            text=body,
            published_at=published_at,
            metadata={
                "query": query,
                "type": "comment" if hit.get("comment_text") else "story",
                "points": hit.get("points"),
                "author": hit.get("author"),
                "num_comments": hit.get("num_comments"),
            },
        )
