"""Pre-compute sentiment scores for all documents and store in SQLite."""

from __future__ import annotations

import logging
import sqlite3

from src.config import DB_PATH
from src.tools.sentiment import classify_batch
from src.utils.logging import setup_logger


logger = setup_logger(__name__)

_BATCH_SIZE = 32

_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_sentiments (
    document_id TEXT PRIMARY KEY,
    title TEXT,
    source TEXT,
    published_at TEXT,
    sentiment_label TEXT NOT NULL,
    sentiment_score REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ds_source ON document_sentiments(source);
CREATE INDEX IF NOT EXISTS idx_ds_published ON document_sentiments(published_at);
"""


def run() -> int:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)

    existing = {
        row["document_id"]
        for row in conn.execute("SELECT document_id FROM document_sentiments").fetchall()
    }

    rows = conn.execute(
        "SELECT id, title, source, published_at FROM documents WHERE title IS NOT NULL"
    ).fetchall()

    to_process = [r for r in rows if r["id"] not in existing]
    if not to_process:
        logger.info("All %d documents already have sentiment scores", len(existing))
        conn.close()
        return 0

    logger.info("Computing sentiment for %d documents", len(to_process))
    titles = [r["title"][:512] for r in to_process]

    all_results = []
    for i in range(0, len(titles), _BATCH_SIZE):
        batch = titles[i : i + _BATCH_SIZE]
        all_results.extend(classify_batch(batch))
        if (i + _BATCH_SIZE) % 100 == 0:
            logger.info("  processed %d / %d", min(i + _BATCH_SIZE, len(titles)), len(titles))

    insert_rows = []
    for row, result in zip(to_process, all_results):
        insert_rows.append((
            row["id"],
            row["title"],
            row["source"],
            row["published_at"] or "",
            result.label,
            result.score,
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO document_sentiments "
        "(document_id, title, source, published_at, sentiment_label, sentiment_score) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        insert_rows,
    )
    conn.commit()
    conn.close()

    logger.info("Stored sentiment for %d documents", len(insert_rows))
    return len(insert_rows)


def main() -> None:
    count = run()
    print(f"\nSentiment indexing complete: {count} documents processed")


if __name__ == "__main__":
    main()
