"""Embedding cache: avoids re-embedding identical queries."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Sequence

from src.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT PRIMARY KEY,
    embedding_json TEXT NOT NULL
);
"""

_initialized = False


def _ensure_table():
    global _initialized
    if _initialized:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    _initialized = True


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def get_cached(text: str) -> list[float] | None:
    _ensure_table()
    text_hash = _hash_text(text)
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT embedding_json FROM embedding_cache WHERE text_hash = ?",
        (text_hash,),
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def put_cached(text: str, embedding: Sequence[float]) -> None:
    _ensure_table()
    text_hash = _hash_text(text)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding_json) VALUES (?, ?)",
        (text_hash, json.dumps(list(embedding))),
    )
    conn.commit()
    conn.close()


def get_or_compute(text: str, compute_fn) -> list[float]:
    """Return cached embedding or compute, cache, and return."""
    cached = get_cached(text)
    if cached is not None:
        return cached
    embedding = compute_fn(text)
    put_cached(text, embedding)
    return embedding
