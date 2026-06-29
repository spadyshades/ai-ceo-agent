"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a uniform stream handler.

    Configuration is applied once to the root logger; repeated calls
    are safe and return module-scoped loggers.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))
        root.addHandler(handler)
        root.setLevel(level)

    # Quiet down chatty third-party loggers
    for noisy in ("urllib3", "chromadb", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger(name)
