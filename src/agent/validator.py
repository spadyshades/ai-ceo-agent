"""Validator node (Phase 5): rule-based + adversarial LLM + source verification."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from src.agent.schema import AgentState, ValidationResult
from src.config import (
    MAX_REPLAN_ATTEMPTS,
    MIN_DISTINCT_SOURCES_PER_RECOMMENDATION,
    MIN_EVIDENCE_PER_RECOMMENDATION,
)
from src.processing.embedder import embed_texts
from src.processing.indexer import get_collection
from src.tools.comparator import find_contradicting_evidence
from src.utils.llm import complete_json


logger = logging.getLogger(__name__)


_ADVERSARIAL_PROMPT = """You are a critical reviewer checking a strategic recommendation for the CEO of BMW.

Recommendation: {title}
Rationale: {rationale}

Contradicting evidence found in the corpus:
{contradictions}

Based on the contradicting evidence, is this recommendation still defensible?

Return a JSON object:
{{
  "defensible": true or false,
  "concern": "one sentence explaining why, or empty string if defensible"
}}

JSON:"""


def _check_rules(state: AgentState) -> list[str]:
    issues: list[str] = []
    if not state.recommendations:
        issues.append("No recommendations were produced")
        return issues

    for i, rec in enumerate(state.recommendations, start=1):
        label = f"rec#{i} '{rec.title[:40]}'"

        if len(rec.evidence_chunk_ids) < MIN_EVIDENCE_PER_RECOMMENDATION:
            issues.append(
                f"{label}: {len(rec.evidence_chunk_ids)} evidence chunks "
                f"(need {MIN_EVIDENCE_PER_RECOMMENDATION})"
            )

        if len(set(rec.evidence_sources)) < MIN_DISTINCT_SOURCES_PER_RECOMMENDATION:
            issues.append(
                f"{label}: {len(set(rec.evidence_sources))} source(s) "
                f"(need {MIN_DISTINCT_SOURCES_PER_RECOMMENDATION})"
            )

        if rec.confidence < 0.2:
            issues.append(f"{label}: very low confidence ({rec.confidence:.2f})")

        if not rec.rationale.strip():
            issues.append(f"{label}: missing rationale")

    return issues


def _check_source_existence(state: AgentState) -> list[str]:
    issues: list[str] = []
    if not state.recommendations:
        return issues

    all_cited: set[str] = set()
    for rec in state.recommendations:
        all_cited.update(rec.evidence_chunk_ids)

    if not all_cited:
        return issues

    try:
        collection = get_collection()
        existing = collection.get(ids=list(all_cited), include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception as exc:
        logger.warning("Source existence check failed: %s", exc)
        return issues

    missing = all_cited - existing_ids
    if missing:
        ratio = len(missing) / len(all_cited)
        if ratio > 0.5:
            issues.append(
                f"{len(missing)}/{len(all_cited)} cited chunk IDs not found in corpus"
            )
    return issues


def _check_claim_alignment(state: AgentState) -> list[str]:
    issues: list[str] = []
    if not state.recommendations:
        return issues

    collection = get_collection()
    for i, rec in enumerate(state.recommendations, start=1):
        if not rec.evidence_chunk_ids:
            continue
        try:
            claim_text = f"{rec.title}. {rec.rationale}"
            claim_emb = np.array(embed_texts([claim_text])[0])

            result = collection.get(
                ids=rec.evidence_chunk_ids, include=["embeddings"]
            )
            valid_ids = result.get("ids", [])
            embeddings = result.get("embeddings")

            if not valid_ids or embeddings is None or len(embeddings) == 0:
                continue

            evidence_embs = np.array(embeddings)
            similarities = np.dot(evidence_embs, claim_emb)
            avg_sim = float(np.mean(similarities))

            if avg_sim < 0.25:
                issues.append(
                    f"rec#{i} '{rec.title[:30]}': weak alignment with cited "
                    f"evidence (avg similarity={avg_sim:.2f})"
                )
        except Exception as exc:
            logger.warning("Claim alignment check failed for rec#%d: %s", i, exc)

    return issues


def _check_adversarial(state: AgentState) -> list[str]:
    issues: list[str] = []
    if not state.recommendations:
        return issues

    for i, rec in enumerate(state.recommendations[:3], start=1):
        try:
            contradictions = find_contradicting_evidence(
                claim=f"{rec.title}. {rec.rationale}", k=3
            )
            if not contradictions:
                continue

            contra_text = "\n".join(
                f"- [{c.source}] {c.title[:80]}: {c.text[:200]}"
                for c in contradictions[:3]
            )

            prompt = _ADVERSARIAL_PROMPT.format(
                title=rec.title,
                rationale=rec.rationale,
                contradictions=contra_text,
            )
            response = complete_json(prompt, temperature=0.1)

            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict) and parsed.get("defensible") is False:
                    concern = parsed.get("concern", "")
                    if concern:
                        issues.append(
                            f"rec#{i} '{rec.title[:30]}': adversarial challenge - {concern}"
                        )
                    else:
                        issues.append(
                            f"rec#{i} '{rec.title[:30]}': failed adversarial review"
                        )
            except json.JSONDecodeError:
                logger.warning("Adversarial response not valid JSON for rec#%d", i)
        except Exception as exc:
            logger.warning("Adversarial check failed for rec#%d: %s", i, exc)

    return issues


def validate_node(state: AgentState) -> dict[str, Any]:
    logger.info("Validator: checking %d recommendations", len(state.recommendations))

    all_issues = _check_rules(state)

    is_final_pass = state.replan_count >= MAX_REPLAN_ATTEMPTS - 1

    if not all_issues and is_final_pass:
        all_issues.extend(_check_source_existence(state))

    if not all_issues and is_final_pass:
        all_issues.extend(_check_claim_alignment(state))

    if not all_issues and is_final_pass:
        all_issues.extend(_check_adversarial(state))

    validation = ValidationResult(passed=len(all_issues) == 0, issues=all_issues)
    logger.info(
        "Validator: %s with %d issue(s)",
        "PASSED" if validation.passed else "FAILED",
        len(all_issues),
    )
    return {"validation": validation}


def route_after_validate(state: AgentState) -> str:
    if state.validation and state.validation.passed:
        return "briefing"
    if state.replan_count >= MAX_REPLAN_ATTEMPTS:
        logger.info(
            "Validator: replan limit reached (%d); proceeding to briefing",
            state.replan_count,
        )
        return "briefing"
    logger.info("Validator: re-planning")
    return "planner"
