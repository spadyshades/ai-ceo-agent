"""Semantic and metadata-filtered retrieval over the BMW corpus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.processing.embedder import embed_texts
from src.processing.indexer import get_collection


@dataclass
class RetrievalHit:
    """A single retrieved chunk with similarity score and metadata."""

    chunk_id: str
    document_id: str
    text: str
    source: str
    url: str
    title: str
    published_at: str
    similarity: float
    metadata: dict[str, Any]


def _normalize_results(results: dict) -> list[RetrievalHit]:
    if not results.get("ids") or not results["ids"][0]:
        return []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[0.0] * len(ids)])[0]

    hits: list[RetrievalHit] = []
    for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
        # Cosine distance to similarity: 1 - distance
        similarity = max(0.0, 1.0 - float(dist))
        hits.append(
            RetrievalHit(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                text=text,
                source=meta.get("source", ""),
                url=meta.get("url", ""),
                title=meta.get("title", ""),
                published_at=meta.get("published_at", ""),
                similarity=similarity,
                metadata=meta,
            )
        )
    return hits


def _build_where(
    source: str | None,
    published_after: datetime | None,
    published_before: datetime | None,
) -> dict[str, Any] | None:
    conditions: list[dict[str, Any]] = []
    if source:
        conditions.append({"source": source})
    if published_after:
        conditions.append({"published_at": {"$gte": published_after.isoformat()}})
    if published_before:
        conditions.append({"published_at": {"$lt": published_before.isoformat()}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search(
    query: str,
    k: int = 5,
    source: str | None = None,
    published_after: datetime | None = None,
    published_before: datetime | None = None,
) -> list[RetrievalHit]:
    """Semantic search with optional metadata filters."""
    collection = get_collection()
    query_embedding = embed_texts([query])[0]

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": k,
    }
    where = _build_where(source, published_after, published_before)
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)
    return _normalize_results(results)


def search_by_entity(
    query: str,
    entity_text: str,
    entity_label: str = "org",
    k: int = 10,
    pre_fetch: int = 30,
) -> list[RetrievalHit]:
    """Search semantically then filter for chunks mentioning a specific entity.

    Args:
        query: Semantic query.
        entity_text: Entity string (case-insensitive substring match).
        entity_label: spaCy label without prefix (org, person, gpe, ...).
        k: Number of hits to return after filtering.
        pre_fetch: How many to retrieve before filtering.
    """
    candidates = search(query, k=pre_fetch)
    needle = entity_text.lower()
    field = f"entities_{entity_label.lower()}"
    filtered = [
        hit for hit in candidates
        if needle in hit.metadata.get(field, "").lower()
    ]
    return filtered[:k]
