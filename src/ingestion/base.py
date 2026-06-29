"""Core types for the ingestion layer."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Document:
    """A single ingested document from any source."""

    id: str
    source: str
    url: str
    title: str
    text: str
    published_at: datetime | None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data = asdict(self)
        data["published_at"] = (
            self.published_at.isoformat() if self.published_at else None
        )
        data["fetched_at"] = self.fetched_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Reconstruct a Document from its dict representation."""
        pub = data.get("published_at")
        fetch = data.get("fetched_at")
        return cls(
            id=data["id"],
            source=data["source"],
            url=data["url"],
            title=data["title"],
            text=data["text"],
            published_at=datetime.fromisoformat(pub) if pub else None,
            fetched_at=(
                datetime.fromisoformat(fetch)
                if fetch
                else datetime.now(timezone.utc)
            ),
            metadata=data.get("metadata", {}),
        )


def make_document_id(source: str, url: str) -> str:
    """Deterministic 16-character document ID derived from source and URL."""
    key = f"{source}::{url}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:16]


class BaseScraper(ABC):
    """Abstract interface implemented by all source scrapers."""

    name: str = ""

    @abstractmethod
    def fetch(self) -> list[Document]:
        """Retrieve documents from the source.

        Implementations should return an empty list rather than raise on
        transient network failures. Hard errors that indicate a code or
        configuration problem should propagate.
        """
