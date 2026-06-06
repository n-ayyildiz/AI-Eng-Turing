"""
Shared utilities for Neurohypothesis v2.

Small, pure functions and one decorator that every module imports rather
than re-implementing. Nothing in here has external side-effects beyond
calling the OpenAI embeddings API (embed_text) and sleeping (with_retries).

Public API:
    - cosine_similarity(a, b) -> float
    - grade_similarity(similarity, context) -> dict
    - embed_text(text) -> list[float]
    - embed_texts(texts) -> list[list[float]]
    - with_retries(fn, max_attempts, backoff_base, exceptions) -> Callable
    - sha256_of(text) -> str
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings
from loguru import logger

import config

# =============================================================================
# Cosine similarity
# =============================================================================

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Returns a value in [-1, 1] — for normalised OpenAI embeddings the
    range is effectively [0, 1].  Returns 0.0 if either vector is zero.

    Used by: originality scoring, gap computation, temporal tagging.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# Three-category grading (shared by originality and gap scoring)
# =============================================================================

def grade_similarity(similarity: float, context: str = "originality") -> dict[str, str]:
    """
    Convert a cosine similarity value into a three-category grade.

    Thresholds are taken from config (VERY_ORIGINAL_THRESHOLD = 0.3,
    LESS_ORIGINAL_THRESHOLD = 0.8) and apply identically to both
    originality scoring and literature-gap scoring — only the display
    labels differ.

    Args:
        similarity: cosine similarity (0 to 1).
        context:    "originality" or "gap" — controls the returned labels.

    Returns:
        Dict with keys 'grade' (very/moderate/less), 'label', 'color'.
    """
    if context == "gap":
        labels = {
            "very":     "Strong gap",
            "moderate": "Moderate gap",
            "less":     "Weak gap",
        }
    else:
        labels = {
            "very":     "Very original",
            "moderate": "Moderately original",
            "less":     "Less original",
        }

    if similarity <= config.VERY_ORIGINAL_THRESHOLD:
        grade = "very"
    elif similarity >= config.LESS_ORIGINAL_THRESHOLD:
        grade = "less"
    else:
        grade = "moderate"

    return {
        "grade": grade,
        "label": labels[grade],
        "color": config.BLUE_GRADE_COLORS[grade],
    }


# =============================================================================
# Embeddings (single shared client, thread-safe initialisation)
# =============================================================================

import threading

_embeddings_client: OpenAIEmbeddings | None = None
_embeddings_lock = threading.Lock()


def _get_embeddings() -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings client (lazy-initialised, thread-safe)."""
    global _embeddings_client
    if _embeddings_client is None:
        with _embeddings_lock:
            if _embeddings_client is None:   # double-checked locking
                _embeddings_client = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    return _embeddings_client


def embed_text(text: str) -> list[float]:
    """
    Embed a single string with text-embedding-3-small.

    Args:
        text: the string to embed (max ~8191 tokens).

    Returns:
        Embedding vector as list[float] of length EMBEDDING_DIMENSIONS.
    """
    return _get_embeddings().embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings in one batched API call.

    Prefer this over calling embed_text() in a loop — one round-trip
    instead of N, and cheaper per-token at high volumes.

    Args:
        texts: list of strings to embed.

    Returns:
        List of embedding vectors, one per input string.
    """
    if not texts:
        return []
    return _get_embeddings().embed_documents(texts)


# =============================================================================
# Retry decorator with exponential backoff
# =============================================================================

def with_retries(
    fn: Callable,
    max_attempts: int = config.PUBMED_MAX_RETRIES,
    backoff_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Return a version of fn that retries up to max_attempts times on failure,
    with exponential backoff between attempts.

    Backoff schedule (backoff_base=2): 1s → 2s → 4s → give up.

    Usage:
        result = with_retries(pubmed_search, max_attempts=3)(query, n=10)

        # Or bind once and reuse:
        safe_search = with_retries(pubmed_search, max_attempts=3)
        result = safe_search(query)

    Args:
        fn:           the callable to wrap.
        max_attempts: total attempts before re-raising the last exception.
        backoff_base: base for the exponential wait (seconds).
        exceptions:   which exception types trigger a retry.

    Returns:
        Wrapped callable with the same signature as fn.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except exceptions as exc:
                if attempt == max_attempts:
                    logger.error(
                        f"{fn.__name__}: all {max_attempts} attempts failed. "
                        f"Last error: {exc}"
                    )
                    raise
                wait = backoff_base ** (attempt - 1)
                logger.warning(
                    f"{fn.__name__}: attempt {attempt}/{max_attempts} failed "
                    f"({exc}). Retrying in {wait:.1f}s…"
                )
                time.sleep(wait)
        raise RuntimeError("unreachable")  # mypy satisfaction

    return wrapper
