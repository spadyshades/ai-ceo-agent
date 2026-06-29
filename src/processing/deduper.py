"""Near-duplicate document detection using MinHash and LSH."""

from __future__ import annotations

import logging
import re
from typing import Iterable

from datasketch import MinHash, MinHashLSH


logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+")


def _shingles(text: str, k: int = 5) -> set[str]:
    """Return word k-shingles for MinHash, lower-cased."""
    tokens = _WORD_RE.findall(text.lower())
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _signature(text: str, num_perm: int) -> MinHash:
    minhash = MinHash(num_perm=num_perm)
    for shingle in _shingles(text):
        minhash.update(shingle.encode("utf-8"))
    return minhash


def find_near_duplicates(
    documents: Iterable[tuple[str, str]],
    similarity_threshold: float = 0.85,
    num_perm: int = 128,
) -> set[str]:
    """Identify near-duplicate documents.

    Args:
        documents: Iterable of (document_id, cleaned_text) pairs.
        similarity_threshold: Jaccard similarity at or above which two
            documents are considered duplicates.
        num_perm: MinHash permutation count.

    Returns:
        Set of document IDs to drop. The first occurrence of each cluster
        is retained.
    """
    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
    duplicates: set[str] = set()

    for doc_id, text in documents:
        signature = _signature(text, num_perm=num_perm)
        if lsh.query(signature):
            duplicates.add(doc_id)
        else:
            lsh.insert(doc_id, signature)

    if duplicates:
        logger.info("Found %d near-duplicate documents", len(duplicates))
    return duplicates
