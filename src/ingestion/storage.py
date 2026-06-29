"""Persistence for ingested documents and ingestion run history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
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


def get_corpus_stats() -> dict:
    """Return summary statistics for the dashboard and CLI."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        per_source = {
            row["source"]: row["count"]
            for row in conn.execute(
                "SELECT source, COUNT(*) AS count FROM documents GROUP BY source"
            )
        }
        last_run = conn.execute(
            "SELECT MAX(run_finished_at) AS last FROM ingestion_log"
        ).fetchone()["last"]
    return {
        "total_documents": total,
        "documents_per_source": per_source,
        "source_count": len(per_source),
        "last_ingestion_at": last_run,
    }
