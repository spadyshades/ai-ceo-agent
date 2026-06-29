"""ChromaDB-backed vector index."""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from src.config import CHROMA_COLLECTION_NAME, CHROMA_DIR


logger = logging.getLogger(__name__)


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Return the corpus collection, creating it with cosine distance if needed."""
    client = get_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    chunk_ids: list[str],
    chunk_texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    """Insert or update chunks in the vector index."""
    if not chunk_ids:
        return
    collection = get_collection()
    collection.upsert(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def collection_stats() -> dict:
    """Return basic collection statistics."""
    collection = get_collection()
    return {
        "name": collection.name,
        "count": collection.count(),
    }
