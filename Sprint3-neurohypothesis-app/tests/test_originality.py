"""
Unit tests for src/engine/originality.py and src/utils.py (cosine + grading).

Tests cover:
    - cosine_similarity: edge cases (zero vectors, orthogonal, identical)
    - grade_similarity: all three grade bands for originality and gap contexts
    - _make_result: correct grade and gate flag from similarity input
    - filter_genuine_gaps: correct filtering with mocked embed_texts
    - score_originality_against_summary: gate flag + grade with mocked embeddings
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config
from src.utils import cosine_similarity, grade_similarity
from src.engine.originality import (
    GapPair,
    OriginalityResult,
    _make_result,
    filter_genuine_gaps,
    score_originality_against_summary,
)


# =============================================================================
# cosine_similarity
# =============================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        # For normalised embeddings this won't occur, but the math should hold
        result = cosine_similarity([1, 0], [-1, 0])
        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_a(self):
        assert cosine_similarity([0, 0], [1, 2]) == 0.0

    def test_zero_vector_b(self):
        assert cosine_similarity([1, 2], [0, 0]) == 0.0

    def test_both_zero(self):
        assert cosine_similarity([0, 0], [0, 0]) == 0.0

    def test_arbitrary_vectors(self):
        # [1,1] and [1,0] → cos = 1/sqrt(2) ≈ 0.707
        result = cosine_similarity([1, 1], [1, 0])
        assert result == pytest.approx(0.7071, abs=1e-3)

    def test_high_dimensional(self):
        v = [0.1] * 1536
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# grade_similarity
# =============================================================================

class TestGradeSimilarity:
    """
    Thresholds: VERY ≤ 0.3, LESS ≥ 0.8, MODERATE in between.
    """

    # ── Originality context ───────────────────────────────────────────────

    def test_very_original(self):
        result = grade_similarity(0.1, context="originality")
        assert result["grade"] == "very"
        assert result["label"] == "Very original"
        assert result["color"] == config.BLUE_GRADE_COLORS["very"]

    def test_moderately_original(self):
        result = grade_similarity(0.5, context="originality")
        assert result["grade"] == "moderate"
        assert result["label"] == "Moderately original"

    def test_less_original(self):
        result = grade_similarity(0.9, context="originality")
        assert result["grade"] == "less"
        assert result["label"] == "Less original"

    def test_boundary_very(self):
        # Exactly at threshold → "very"
        result = grade_similarity(config.VERY_ORIGINAL_THRESHOLD, "originality")
        assert result["grade"] == "very"

    def test_boundary_less(self):
        # Exactly at threshold → "less"
        result = grade_similarity(config.LESS_ORIGINAL_THRESHOLD, "originality")
        assert result["grade"] == "less"

    # ── Gap context ───────────────────────────────────────────────────────

    def test_strong_gap(self):
        result = grade_similarity(0.1, context="gap")
        assert result["grade"] == "very"
        assert result["label"] == "Strong gap"

    def test_moderate_gap(self):
        result = grade_similarity(0.5, context="gap")
        assert result["grade"] == "moderate"
        assert result["label"] == "Moderate gap"

    def test_weak_gap(self):
        result = grade_similarity(0.9, context="gap")
        assert result["grade"] == "less"
        assert result["label"] == "Weak gap"

    def test_unknown_context_defaults_to_originality_labels(self):
        result = grade_similarity(0.2, context="other")
        assert result["label"] == "Very original"


# =============================================================================
# _make_result
# =============================================================================

class TestMakeResult:
    def test_passes_gate_when_original(self):
        hyp = {"id": "H1", "statement": "dopaminergic modulation of hippocampal neurogenesis"}
        result = _make_result(hyp, similarity=0.1)
        assert result.passes_gate is True
        assert result.originality_score == pytest.approx(0.9, abs=1e-6)
        assert result.grade == "very"

    def test_fails_gate_when_too_similar(self):
        hyp = {"id": "H1", "statement": "restatement of existing work"}
        result = _make_result(hyp, similarity=0.85)
        assert result.passes_gate is False
        assert result.originality_score == pytest.approx(0.15, abs=1e-6)
        assert result.grade == "less"

    def test_boundary_passes(self):
        # originality_score = 1 - 0.75 = 0.25 > 0.2 → passes
        hyp = {"id": "H1", "statement": "test"}
        result = _make_result(hyp, similarity=0.75)
        assert result.passes_gate is True

    def test_boundary_fails(self):
        # originality_score = 1 - 0.82 = 0.18 < 0.2 → fails
        hyp = {"id": "H1", "statement": "test"}
        result = _make_result(hyp, similarity=0.82)
        assert result.passes_gate is False

    def test_hypothesis_text_stored(self):
        hyp = {"id": "H2", "statement": "APOE-ε4 and amyloid accumulation"}
        result = _make_result(hyp, similarity=0.3)
        assert result.hypothesis_text == hyp["statement"]
        assert result.hypothesis_id == "H2"


# =============================================================================
# filter_genuine_gaps
# =============================================================================

class TestFilterGenuineGaps:
    """Uses mocked embed_texts so no API calls are made."""

    def _mock_embeddings(self):
        return MagicMock()

    def test_empty_inputs_return_empty(self):
        emb = self._mock_embeddings()
        result = filter_genuine_gaps([], [], emb)
        assert result == []

    def test_empty_past_returns_empty(self):
        emb = self._mock_embeddings()
        future = [{"text": "future work", "paper_id": "p1"}]
        result = filter_genuine_gaps([], future, emb)
        assert result == []

    @patch("src.engine.originality.embed_texts")
    def test_orthogonal_vectors_are_genuine_gaps(self, mock_embed):
        # past and future embeddings are orthogonal → sim=0 → big gap
        mock_embed.side_effect = [[[1, 0]], [[0, 1]]]
        emb = MagicMock()
        past   = [{"text": "prior fMRI findings", "paper_id": "p1"}]
        future = [{"text": "recommended future EEG work", "paper_id": "p2"}]

        result = filter_genuine_gaps(past, future, emb, threshold=0.8)
        assert len(result) == 1
        assert result[0].gap_score == pytest.approx(1.0, abs=1e-6)
        assert result[0].past_paper_id  == "p1"
        assert result[0].future_paper_id == "p2"

    @patch("src.engine.originality.embed_texts")
    def test_identical_vectors_filtered_out(self, mock_embed):
        # past and future embeddings are identical → sim=1 → not a gap
        mock_embed.side_effect = [[[1, 0]], [[1, 0]]]
        emb = MagicMock()
        past   = [{"text": "same finding restated", "paper_id": "p1"}]
        future = [{"text": "same finding restated", "paper_id": "p1"}]

        result = filter_genuine_gaps(past, future, emb, threshold=0.8)
        assert result == []

    @patch("src.engine.originality.embed_texts")
    def test_sorted_by_gap_strength(self, mock_embed):
        # Two past chunks vs one future — stronger gap should come first
        mock_embed.side_effect = [
            [[1, 0], [0.9, 0.1]],   # past vecs
            [[0, 1]],               # future vec
        ]
        emb = MagicMock()
        past = [
            {"text": "past A", "paper_id": "pA"},
            {"text": "past B", "paper_id": "pB"},
        ]
        future = [{"text": "future X", "paper_id": "pX"}]

        result = filter_genuine_gaps(past, future, emb, threshold=0.8)
        assert len(result) == 2
        assert result[0].gap_score >= result[1].gap_score


# =============================================================================
# score_originality_against_summary
# =============================================================================

class TestScoreOriginalityAgainstSummary:

    @patch("src.engine.originality.embed_texts")
    def test_passes_when_dissimilar(self, mock_embed):
        # hypothesis vec orthogonal to past_summary → originality=1.0
        mock_embed.return_value = [[1, 0], [0, 1]]  # [hyp_vec, past_vec]
        emb = MagicMock()
        hyps = [{"id": "H1", "statement": "novel hypothesis about APOE"}]

        results = score_originality_against_summary(hyps, "past summary text", emb)
        assert len(results) == 1
        assert results[0].originality_score == pytest.approx(1.0, abs=1e-6)
        assert results[0].passes_gate is True

    @patch("src.engine.originality.embed_texts")
    def test_fails_when_similar(self, mock_embed):
        # hypothesis nearly identical to past_summary → originality ≈ 0
        mock_embed.return_value = [[1, 0], [1, 0]]
        emb = MagicMock()
        hyps = [{"id": "H1", "statement": "restatement"}]

        results = score_originality_against_summary(hyps, "past summary", emb)
        assert results[0].originality_score == pytest.approx(0.0, abs=1e-6)
        assert results[0].passes_gate is False

    def test_empty_past_summary_all_original(self):
        emb  = MagicMock()
        hyps = [{"id": "H1", "statement": "some hypothesis"}]
        results = score_originality_against_summary(hyps, "", emb)
        assert results[0].originality_score == 1.0
        assert results[0].passes_gate is True

    def test_empty_hypotheses_returns_empty(self):
        emb = MagicMock()
        assert score_originality_against_summary([], "past summary", emb) == []
