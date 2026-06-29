"""Find supporting and contradicting evidence for a claim."""

from __future__ import annotations

from src.tools.retriever import RetrievalHit, search
from src.utils.llm import complete


_NEGATION_PROMPT = (
    "Rewrite the following claim as its direct opposite. "
    "Return only the negated claim, with no additional text, "
    "no quotes, and no explanation.\n\n"
    "Claim: {claim}\n\n"
    "Negated claim:"
)


def find_supporting_evidence(claim: str, k: int = 5) -> list[RetrievalHit]:
    """Retrieve corpus passages that semantically support the claim."""
    return search(query=claim, k=k)


def find_contradicting_evidence(claim: str, k: int = 5) -> list[RetrievalHit]:
    """Retrieve corpus passages likely to contradict the claim.

    The claim is first negated via the LLM; chunks most similar to the
    negation are returned as candidate contradictions.
    """
    prompt = _NEGATION_PROMPT.format(claim=claim)
    negation = complete(prompt, temperature=0.1).strip()
    if not negation:
        return []
    return search(query=negation, k=k)


def negate_claim(claim: str) -> str:
    """Expose the negation step on its own for explainability."""
    return complete(_NEGATION_PROMPT.format(claim=claim), temperature=0.1).strip()
