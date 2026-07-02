"""Semantic, BM25, and hybrid retrieval over the BMW corpus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi

from src.processing.embedder import embed_texts
from src.processing.indexer import get_collection


@dataclass
class RetrievalHit:
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
    """Search semantically then filter for chunks mentioning a specific entity."""
    candidates = search(query, k=pre_fetch)
    needle = entity_text.lower()
    field = f"entities_{entity_label.lower()}"
    filtered = [
        hit for hit in candidates
        if needle in hit.metadata.get(field, "").lower()
    ]
    return filtered[:k]


# --- BM25 keyword search ---

@lru_cache(maxsize=1)
def _build_bm25_index() -> tuple[BM25Okapi, list[str], list[dict]]:
    """Build a BM25 index over all chunks in the collection."""
    collection = get_collection()
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data.get("ids", [])
    docs = all_data.get("documents", [])
    metas = all_data.get("metadatas", [])

    tokenized = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, ids, metas


def search_bm25(query: str, k: int = 5) -> list[RetrievalHit]:
    """Keyword-based BM25 search over all chunks."""
    bm25, ids, metas = _build_bm25_index()
    collection = get_collection()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored = sorted(
        zip(range(len(ids)), scores), key=lambda x: x[1], reverse=True
    )[:k]

    all_docs = collection.get(include=["documents", "metadatas"])
    docs = all_docs.get("documents", [])

    hits: list[RetrievalHit] = []
    max_score = scored[0][1] if scored and scored[0][1] > 0 else 1.0
    for idx, score in scored:
        if score <= 0:
            continue
        meta = metas[idx] if idx < len(metas) else {}
        text = docs[idx] if idx < len(docs) else ""
        hits.append(
            RetrievalHit(
                chunk_id=ids[idx],
                document_id=meta.get("document_id", ""),
                text=text,
                source=meta.get("source", ""),
                url=meta.get("url", ""),
                title=meta.get("title", ""),
                published_at=meta.get("published_at", ""),
                similarity=float(score / max_score),
                metadata=meta,
            )
        )
    return hits


def search_hybrid(
    query: str,
    k: int = 5,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3,
    pre_fetch: int = 20,
) -> list[RetrievalHit]:
    """Reciprocal rank fusion of semantic and BM25 search."""
    semantic_hits = search(query, k=pre_fetch)
    bm25_hits = search_bm25(query, k=pre_fetch)

    rrf_scores: dict[str, float] = {}
    hit_map: dict[str, RetrievalHit] = {}

    for rank, hit in enumerate(semantic_hits):
        rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0) + (
            semantic_weight / (rank + 60)
        )
        hit_map[hit.chunk_id] = hit

    for rank, hit in enumerate(bm25_hits):
        rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0) + (
            bm25_weight / (rank + 60)
        )
        if hit.chunk_id not in hit_map:
            hit_map[hit.chunk_id] = hit

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results: list[RetrievalHit] = []
    max_rrf = ranked[0][1] if ranked else 1.0
    for chunk_id, rrf_score in ranked:
        hit = hit_map[chunk_id]
        hit.similarity = round(rrf_score / max_rrf, 4)
        results.append(hit)
    return results
