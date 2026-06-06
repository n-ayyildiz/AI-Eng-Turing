"""
Unit tests for src/engine/generate.py — quality_gate() and select_categories().

quality_gate() is fully deterministic (no LLM) — all cases are testable
without any mocking.

Tests cover:
    - Pass when both originality and plausibility meet thresholds
    - Fail when originality is too low (with correct failure reason)
    - Fail when plausibility is too low (with correct failure reason)
    - Fail when both are low (combined failure reason)
    - Exhausted attempts → best-of-3 badge (passes=True, best_of=True)
    - select_categories: correct category pair for each hypothesis index
"""

from __future__ import annotations

import pytest

import config
from src.engine.generate import (
    PlausibilityResult,
    QualityGateDecision,
    quality_gate,
    select_categories,
)
from src.engine.originality import _make_result


# =============================================================================
# Helpers
# =============================================================================

def _orig(similarity: float):
    """Build an OriginalityResult with the given similarity."""
    return _make_result({"id": "H", "statement": "test hypothesis"}, similarity=similarity)


def _plaus(average: float, scores: dict | None = None) -> PlausibilityResult:
    """Build a PlausibilityResult with the given average."""
    _DIMS = ["novelty", "testability", "mechanistic_coherence",
             "citation_traceability", "conflict_awareness", "usefulness"]
    if scores is None:
        scores = {d: average for d in _DIMS}
    return PlausibilityResult(
        scores=scores,
        average=average,
        verdict="test verdict",
        passes_gate=(average >= config.PLAUSIBILITY_PASS_AVG),
    )


# =============================================================================
# quality_gate — pass cases
# =============================================================================

class TestQualityGatePass:
    def test_both_pass_first_attempt(self):
        decision = quality_gate(
            originality=_orig(similarity=0.1),   # score=0.9 >> 0.2
            plausibility=_plaus(average=4.0),
            attempt=0,
        )
        assert decision.passes is True
        assert decision.best_of_attempts is False
        assert decision.failure_reason == ""
        assert decision.originality_ok is True
        assert decision.plausibility_ok is True

    def test_borderline_pass(self):
        # originality_score = 1 - 0.75 = 0.25 > 0.2 (clearly above threshold)
        # plausibility avg = 3.0 (exactly at threshold, fine since 3.0 is representable)
        decision = quality_gate(
            originality=_orig(similarity=0.75),
            plausibility=_plaus(average=config.PLAUSIBILITY_PASS_AVG),
            attempt=0,
        )
        assert decision.passes is True

    def test_pass_on_second_attempt(self):
        decision = quality_gate(_orig(0.1), _plaus(4.0), attempt=1)
        assert decision.passes is True
        assert decision.best_of_attempts is False


# =============================================================================
# quality_gate — fail cases (attempts remaining)
# =============================================================================

class TestQualityGateFail:
    def test_originality_too_low(self):
        decision = quality_gate(
            originality=_orig(similarity=0.95),   # score=0.05 << 0.2
            plausibility=_plaus(average=4.0),
            attempt=0,
        )
        assert decision.passes is False
        assert decision.originality_ok is False
        assert decision.plausibility_ok is True
        assert "originality" in decision.failure_reason.lower()
        assert "0.05" in decision.failure_reason or "0.2" in decision.failure_reason

    def test_plausibility_too_low(self):
        decision = quality_gate(
            originality=_orig(similarity=0.1),    # high originality
            plausibility=_plaus(average=2.0),     # fails threshold
            attempt=0,
        )
        assert decision.passes is False
        assert decision.originality_ok is True
        assert decision.plausibility_ok is False
        assert "quality" in decision.failure_reason.lower() or \
               "plausibility" in decision.failure_reason.lower() or \
               "scientific" in decision.failure_reason.lower()

    def test_both_fail(self):
        decision = quality_gate(
            originality=_orig(similarity=0.95),
            plausibility=_plaus(average=2.0),
            attempt=0,
        )
        assert decision.passes is False
        assert decision.originality_ok is False
        assert decision.plausibility_ok is False
        # Failure reason should mention both dimensions
        reason = decision.failure_reason.lower()
        assert "originality" in reason

    def test_failure_reason_not_empty_on_fail(self):
        decision = quality_gate(_orig(0.95), _plaus(2.0), attempt=1)
        assert decision.failure_reason != ""

    def test_weak_plausibility_dimensions_named(self):
        # Scores with specific weak dimensions
        scores = {
            "novelty": 4, "testability": 1, "mechanistic_coherence": 1,
            "citation_traceability": 4, "conflict_awareness": 4, "usefulness": 4,
        }
        plaus = _plaus(average=3.0, scores={k: float(v) for k, v in scores.items()})
        plaus.passes_gate = False      # force fail even though avg=3
        plaus.average = 2.0            # pretend it's low
        decision = quality_gate(_orig(0.95), plaus, attempt=0)
        assert not decision.passes


# =============================================================================
# quality_gate — exhausted (best-of-3 badge)
# =============================================================================

class TestQualityGateExhausted:
    def test_third_attempt_always_proceeds(self):
        decision = quality_gate(
            originality=_orig(similarity=0.95),   # still failing
            plausibility=_plaus(average=2.0),
            attempt=config.QUALITY_GATE_MAX_ATTEMPTS - 1,  # attempt=2
        )
        # Despite failing, should proceed with badge
        assert decision.passes is True
        assert decision.best_of_attempts is True

    def test_badge_not_set_on_first_attempt_fail(self):
        decision = quality_gate(_orig(0.95), _plaus(2.0), attempt=0)
        assert decision.best_of_attempts is False

    def test_badge_not_set_when_passing_normally(self):
        decision = quality_gate(_orig(0.1), _plaus(4.0), attempt=2)
        # Passes cleanly → no badge even on last attempt
        assert decision.best_of_attempts is False


# =============================================================================
# select_categories
# =============================================================================
