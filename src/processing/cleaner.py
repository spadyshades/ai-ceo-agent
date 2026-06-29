"""Text cleaning utilities for ingested documents."""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup


_WHITESPACE_RE = re.compile(r"\s+")

_BOILERPLATE_PATTERNS = [
    re.compile(
        r"^(cookie|privacy|gdpr|accept|reject).*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"subscribe to our newsletter.*$", re.IGNORECASE | re.MULTILINE
    ),
    re.compile(r"share this article.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"read more.*$", re.IGNORECASE | re.MULTILINE),
]


def strip_html(text: str) -> str:
    """Remove residual HTML markup."""
    if "<" not in text:
        return text
    return BeautifulSoup(text, "lxml").get_text(separator=" ", strip=True)


def normalize_unicode(text: str) -> str:
    """Normalise unicode to NFKC form."""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def remove_boilerplate(text: str) -> str:
    """Strip common boilerplate lines."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return text


def clean(text: str) -> str:
    """Apply the full cleaning pipeline."""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    return text
