"""
Tests for the v2.1 post-retrieval relevance check.

Verifies:
- per-abstract cosine threshold gate
- min_pass count behavior
- mean_cosine reporting
- pass/fail signal under various scenarios
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _FakePaper:
    """Minimal stand-in for tools.pubmed.PubMedPaper."""
    pmid:           str
    title:          str             = ""
    abstract:       str             = ""
    journal:        str             = ""
    year:           int             = 2024
    query_cosine:   float | None    = None


def test_passes_when_min_pass_papers_clear_threshold() -> None:
    from src.tools.pubmed import check_retrieval_relevance

    papers = [
        _FakePaper(pmid=str(i), query_cosine=c)
        for i, c in enumerate([0.50, 0.40, 0.35, 0.30, 0.27, 0.20, 0.15, 0.10, 0.05, 0.01])
    ]
    passed, n_pass, kept, mean_cos = check_retrieval_relevance(
        papers, threshold=0.25, min_pass=5,
    )
    assert passed is True
    assert n_pass == 5
    assert len(kept) == 5
    assert mean_cos == pytest.approx(0.233, abs=0.01)


def test_fails_when_below_min_pass() -> None:
    from src.tools.pubmed import check_retrieval_relevance

    papers = [
        _FakePaper(pmid=str(i), query_cosine=c)
        for i, c in enumerate([0.30, 0.28, 0.27, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0])
    ]
    passed, n_pass, kept, _ = check_retrieval_relevance(
        papers, threshold=0.25, min_pass=5,
    )
    assert passed is False
    assert n_pass == 3
    assert len(kept) == 3


def test_empty_papers() -> None:
    from src.tools.pubmed import check_retrieval_relevance

    passed, n_pass, kept, mean_cos = check_retrieval_relevance(
        [], threshold=0.25, min_pass=5,
    )
    assert passed is False
    assert n_pass == 0
    assert kept == []
    assert mean_cos == 0.0


def test_all_below_threshold() -> None:
    from src.tools.pubmed import check_retrieval_relevance

    papers = [_FakePaper(pmid=str(i), query_cosine=0.10) for i in range(10)]
    passed, n_pass, kept, _ = check_retrieval_relevance(
        papers, threshold=0.25, min_pass=5,
    )
    assert passed is False
    assert n_pass == 0
    assert kept == []


def test_all_above_threshold() -> None:
    from src.tools.pubmed import check_retrieval_relevance

    papers = [_FakePaper(pmid=str(i), query_cosine=0.50) for i in range(10)]
    passed, n_pass, kept, mean_cos = check_retrieval_relevance(
        papers, threshold=0.25, min_pass=5,
    )
    assert passed is True
    assert n_pass == 10
    assert len(kept) == 10
    assert mean_cos == 0.5
