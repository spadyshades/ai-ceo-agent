"""Persistence for ingested documents and ingestion run history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from src.config import DB_PATH, RAW_DIR
from src.ingestion.base import Document


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    run_started_at TEXT NOT NULL,
    run_finished_at TEXT NOT NULL,
    documents_fetched INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    file_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_fetched_at ON documents(fetched_at);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    indexed_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with the schema applied."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_documents(documents: list[Document]) -> int:
    """Persist documents as JSON files and register them in SQLite.

    Returns the number of newly stored documents (excluding duplicates
    already present in the database).
    """
    if not documents:
        return 0

    new_count = 0
    with get_connection() as conn:
        for doc in documents:
            existing = conn.execute(
                "SELECT 1 FROM documents WHERE id = ?", (doc.id,)
            ).fetchone()
            if existing:
                continue

            source_dir = RAW_DIR / doc.source
            source_dir.mkdir(parents=True, exist_ok=True)
            file_path = source_dir / f"{doc.id}.json"
            file_path.write_text(
                json.dumps(doc.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            conn.execute(
                """
                INSERT INTO documents
                (id, source, url, title, published_at, fetched_at, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.id,
                    doc.source,
                    doc.url,
                    doc.title,
                    doc.published_at.isoformat() if doc.published_at else None,
                    doc.fetched_at.isoformat(),
                    str(file_path),
                ),
            )
            new_count += 1

    return new_count


def log_ingestion_run(
    source: str,
    started_at: datetime,
    finished_at: datetime,
    documents_fetched: int,
    status: str = "success",
    error: str | None = None,
) -> None:
    """Append a row to the ingestion_log table."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO ingestion_log
            (source, run_started_at, run_finished_at, documents_fetched, status, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                started_at.isoformat(),
                finished_at.isoformat(),
                documents_fetched,
                status,
                error,
            ),
        )


def iter_unprocessed_documents(include_all: bool = False) -> Iterator[Document]:
    """Yield Documents that have not yet been chunked and indexed.

    Args:
        include_all: If True, yield every document regardless of indexing state.
    """
    with get_connection() as conn:
        if include_all:
            rows = conn.execute("SELECT file_path FROM documents").fetchall()
        else:
            rows = conn.execute(
                """
                SELECT d.file_path FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE c.id IS NULL
                """
            ).fetchall()

    for row in rows:
        file_path = Path(row["file_path"])
        if not file_path.exists():
            continue
        data = json.loads(file_path.read_text(encoding="utf-8"))
        yield Document.from_dict(data)


def mark_chunks_indexed(rows: list[tuple]) -> None:
    """Record indexed chunks in SQLite.

    Args:
        rows: List of (chunk_id, document_id, chunk_index, char_count,
            indexed_at_iso) tuples.
    """
    if not rows:
        return
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (id, document_id, chunk_index, char_count, indexed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


def get_corpus_stats() -> dict:
    """Return corpus and index summary statistics."""
    with get_connection() as conn:
        total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        per_source = {
            row["source"]: row["count"]
            for row in conn.execute(
                "SELECT source, COUNT(*) AS count FROM documents GROUP BY source"
            )
        }
        last_run = conn.execute(
            "SELECT MAX(run_finished_at) AS last FROM ingestion_log"
        ).fetchone()["last"]
        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        indexed_docs = conn.execute(
            "SELECT COUNT(DISTINCT document_id) FROM chunks"
        ).fetchone()[0]

    return {
        "total_documents": total_docs,
        "documents_per_source": per_source,
        "source_count": len(per_source),
        "last_ingestion_at": last_run,
        "total_chunks": total_chunks,
        "indexed_documents": indexed_docs,
    }
