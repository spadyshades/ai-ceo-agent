"""End-to-end processing pipeline: clean, dedupe, extract, chunk, embed, index."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from src.ingestion.base import Document
from src.ingestion.storage import (
    get_corpus_stats,
    iter_unprocessed_documents,
    mark_chunks_indexed,
)
from src.processing.chunker import chunk_text
from src.processing.cleaner import clean
from src.processing.deduper import find_near_duplicates
from src.processing.embedder import embed_texts
from src.processing.extractor import extract_entities
from src.processing.indexer import collection_stats, upsert_chunks
from src.utils.logging import setup_logger
from src.processing.sentiment_indexer import run as run_sentiment


logger = setup_logger(__name__)


_EMBED_BATCH_SIZE = 32
_DEDUP_SIMILARITY_THRESHOLD = 0.85
_ENTITY_LIST_CAP = 20


def _build_chunk_metadata(
    doc: Document,
    chunk_index: int,
    total_chunks: int,
    entities: dict[str, list[str]],
) -> dict[str, Any]:
    """Construct chunk metadata for ChromaDB. Values must be primitives."""
    metadata: dict[str, Any] = {
        "document_id": doc.id,
        "source": doc.source,
        "url": doc.url,
        "title": doc.title,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "published_at": doc.published_at.isoformat() if doc.published_at else "",
    }
    for label, values in entities.items():
        metadata[f"entities_{label.lower()}"] = ";".join(values[:_ENTITY_LIST_CAP])
    return metadata


def process_all(force_reprocess: bool = False) -> dict[str, int]:
    """Run the processing pipeline over all unindexed documents."""
    documents = list(iter_unprocessed_documents(include_all=force_reprocess))
    logger.info("Loaded %d documents for processing", len(documents))

    if not documents:
        return {
            "documents_in": 0,
            "duplicates_dropped": 0,
            "documents_indexed": 0,
            "chunks_indexed": 0,
        }

    # Clean
    cleaned_texts: dict[str, str] = {doc.id: clean(doc.text) for doc in documents}

    # Dedupe at document level
    duplicate_ids = find_near_duplicates(
        list(cleaned_texts.items()),
        similarity_threshold=_DEDUP_SIMILARITY_THRESHOLD,
    )
    documents = [d for d in documents if d.id not in duplicate_ids]
    logger.info(
        "After dedup: %d documents retained, %d dropped",
        len(documents),
        len(duplicate_ids),
    )

    # Extract entities and chunk
    chunk_records: list[dict[str, Any]] = []
    for doc in documents:
        cleaned = cleaned_texts[doc.id]
        if not cleaned:
            continue
        entities = extract_entities(cleaned)
        chunks = chunk_text(cleaned)
        for chunk in chunks:
            chunk_id = f"{doc.id}_c{chunk.index:03d}"
            chunk_records.append(
                {
                    "id": chunk_id,
                    "doc": doc,
                    "text": chunk.text,
                    "chunk_index": chunk.index,
                    "total_chunks": len(chunks),
                    "entities": entities,
                }
            )

    logger.info("Generated %d chunks", len(chunk_records))
    if not chunk_records:
        return {
            "documents_in": len(documents),
            "duplicates_dropped": len(duplicate_ids),
            "documents_indexed": 0,
            "chunks_indexed": 0,
        }

    # Embed
    chunk_texts = [r["text"] for r in chunk_records]
    logger.info("Embedding %d chunks", len(chunk_texts))
    embeddings = embed_texts(chunk_texts, batch_size=_EMBED_BATCH_SIZE)

    # Upsert to Chroma
    chunk_ids = [r["id"] for r in chunk_records]
    metadatas = [
        _build_chunk_metadata(
            r["doc"], r["chunk_index"], r["total_chunks"], r["entities"]
        )
        for r in chunk_records
    ]
    upsert_chunks(
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("Upserted %d chunks into Chroma collection", len(chunk_ids))

    # Record chunks in SQLite
    indexed_at_iso = datetime.now(timezone.utc).isoformat()
    chunk_rows = [
        (
            r["id"],
            r["doc"].id,
            r["chunk_index"],
            len(r["text"]),
            indexed_at_iso,
        )
        for r in chunk_records
    ]
    mark_chunks_indexed(chunk_rows)

    indexed_doc_ids = {r["doc"].id for r in chunk_records}

    return {
        "documents_in": len(documents) + len(duplicate_ids),
        "duplicates_dropped": len(duplicate_ids),
        "documents_indexed": len(indexed_doc_ids),
        "chunks_indexed": len(chunk_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the processing pipeline")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Reprocess all documents, including those already indexed",
    )
    args = parser.parse_args()

    summary = process_all(force_reprocess=args.reprocess)
    corpus = get_corpus_stats()
    index = collection_stats()
    sentiment_count = run_sentiment()
    logger.info("Pre-computed sentiment for %d documents", sentiment_count)

    print("\nProcessing summary")
    print("-" * 60)
    print(f"  Documents loaded for run    {summary['documents_in']:>5}")
    print(f"  Near-duplicates dropped     {summary['duplicates_dropped']:>5}")
    print(f"  Documents indexed this run  {summary['documents_indexed']:>5}")
    print(f"  Chunks indexed this run     {summary['chunks_indexed']:>5}")
    print("-" * 60)
    print(f"  Corpus total documents      {corpus['total_documents']:>5}")
    print(f"  Corpus indexed documents    {corpus['indexed_documents']:>5}")
    print(f"  Total chunks in index       {corpus['total_chunks']:>5}")
    print(f"  Chroma collection count     {index['count']:>5}")


if __name__ == "__main__":
    main()
