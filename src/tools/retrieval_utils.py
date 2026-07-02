"""Query-time utilities for retrieval results."""

from __future__ import annotations

from src.tools.retriever import RetrievalHit


def dedupe_by_document(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Keep only the highest-scoring chunk per document_id."""
    seen: dict[str, RetrievalHit] = {}
    for hit in hits:
        doc_id = hit.document_id
        if doc_id not in seen or hit.similarity > seen[doc_id].similarity:
            seen[doc_id] = hit
    deduped = sorted(seen.values(), key=lambda h: h.similarity, reverse=True)
    return deduped
