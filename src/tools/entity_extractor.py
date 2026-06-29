"""Tool-layer wrapper around the spaCy entity extractor."""

from __future__ import annotations

from src.processing.extractor import extract_entities


def extract(text: str) -> dict[str, list[str]]:
    """Return named entities grouped by label."""
    return extract_entities(text)


def extract_organizations(text: str) -> list[str]:
    return extract(text).get("ORG", [])


def extract_locations(text: str) -> list[str]:
    return extract(text).get("GPE", [])


def extract_people(text: str) -> list[str]:
    return extract(text).get("PERSON", [])


def extract_products(text: str) -> list[str]:
    return extract(text).get("PRODUCT", [])
