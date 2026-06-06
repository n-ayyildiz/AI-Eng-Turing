"""
Input validation for Neurohypothesis v2 (N1 validate_input).

Three-layer guard applied before any LLM or PubMed call:

    Layer 1 — Length check (free, instant)
        Rejects inputs shorter than MIN_QUERY_LENGTH or longer than
        MAX_QUERY_LENGTH.

    Layer 2 — Session throttle (free, instant)
        Rejects a new run if one started less than THROTTLE_MIN_SECONDS
        ago in the same browser session.  Prevents rapid button-mashing
        and accidental double-submissions.  Addresses reviewer comment #2.

    Layer 3 — OpenAI Moderation API (free endpoint, ~100 ms)
        Flags inputs containing hate speech, violence, sexual content,
        self-harm, etc.  Replaces the v1 regex-only guard (reviewer #1).
        The regex approach is kept as a cheap pre-filter for obviously
        malicious prompts to avoid even the Moderation API call.

On any failure the pipeline receives a ModerationResult with passed=False
and a user-facing reason string.  The graph's ERR terminal is reached by
Decision A checking result.passed.

Public API:
    - ModerationResult     dataclass
    - validate_input(topic, session_state) -> ModerationResult
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from loguru import logger
from openai import OpenAI

import config

# =============================================================================
# Configuration
# =============================================================================

THROTTLE_MIN_SECONDS = 10     # minimum gap between pipeline runs per session
SESSION_TIMESTAMP_KEY = "_neurohypothesis_last_run_ts"

# Cheap regex pre-filter — catches only the most obvious injection attempts
# so we don't waste an API call on them.  Not a security boundary by itself.
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+instructions?"
    r"|<\s*/?script"
    r"|system\s*:\s*(you\s+are|act\s+as)"
    r"|\bprompt\s+injection\b)",
    re.IGNORECASE,
)


# =============================================================================
# Result type
# =============================================================================

@dataclass
class ModerationResult:
    """Outcome of validate_input()."""
    passed:          bool
    reason:          str            # empty string when passed=True
    flagged_by:      str            # "length" | "throttle" | "regex" | "moderation" | ""
    categories:      dict[str, bool] = field(default_factory=dict)
                                    # OpenAI moderation category flags (empty on pass)
    moderation_used: bool            = False  # False if API was not reached


# =============================================================================
# Main entry point
# =============================================================================

def validate_input(
    topic:          str,
    session_state:  dict,
) -> ModerationResult:
    """
    Run the three-layer guard on the user's raw topic string.

    Args:
        topic:          the raw text from the Streamlit input box.
        session_state:  st.session_state dict — used for throttle timestamp.

    Returns:
        ModerationResult with passed=True if all layers clear.
    """
    # ── Layer 1: length ───────────────────────────────────────────────────
    stripped = topic.strip()

    if len(stripped) < config.MIN_QUERY_LENGTH:
        return ModerationResult(
            passed=False,
            reason=f"Topic too short — please enter at least {config.MIN_QUERY_LENGTH} characters.",
            flagged_by="length",
        )

    if len(stripped) > config.MAX_QUERY_LENGTH:
        return ModerationResult(
            passed=False,
            reason=(
                f"Topic too long ({len(stripped)} characters). "
                f"Please keep it under {config.MAX_QUERY_LENGTH} characters."
            ),
            flagged_by="length",
        )

    # ── Layer 2: session throttle ─────────────────────────────────────────
    last_run = session_state.get(SESSION_TIMESTAMP_KEY, 0.0)
    elapsed  = time.time() - last_run

    if elapsed < THROTTLE_MIN_SECONDS:
        wait = int(THROTTLE_MIN_SECONDS - elapsed) + 1
        logger.warning(f"Throttle: run blocked ({elapsed:.1f}s since last run, need {THROTTLE_MIN_SECONDS}s)")
        return ModerationResult(
            passed=False,
            reason=f"Please wait {wait} seconds before starting a new run.",
            flagged_by="throttle",
        )

    # ── Layer 3a: cheap regex pre-filter ──────────────────────────────────
    if _INJECTION_PATTERNS.search(stripped):
        logger.warning(f"Regex pre-filter triggered on: {stripped[:80]!r}")
        return ModerationResult(
            passed=False,
            reason="Input appears to contain an instruction override attempt. Please enter a genuine neuroscience research topic.",
            flagged_by="regex",
        )

    # ── Layer 3b: OpenAI Moderation API ───────────────────────────────────
    try:
        client   = OpenAI()
        response = client.moderations.create(input=stripped)
        result   = response.results[0]

        if result.flagged:
            # Build a human-readable list of triggered categories
            triggered = [
                cat.replace("_", " ").replace("/", " / ")
                for cat, flagged in result.categories.__dict__.items()
                if flagged
            ]
            logger.warning(
                f"Moderation API flagged input. Categories: {triggered}. "
                f"Input: {stripped[:80]!r}"
            )
            return ModerationResult(
                passed=False,
                reason=(
                    "Input was flagged by the content moderation system "
                    f"({', '.join(triggered)}). "
                    "Please enter a genuine neuroscience research topic."
                ),
                flagged_by="moderation",
                categories=result.categories.__dict__,
                moderation_used=True,
            )

        logger.info(f"Input validation passed | topic: {stripped[:60]!r}")
        # Record successful run timestamp for throttle
        session_state[SESSION_TIMESTAMP_KEY] = time.time()

        return ModerationResult(
            passed=True,
            reason="",
            flagged_by="",
            moderation_used=True,
        )

    except Exception as exc:
        # Moderation API unavailable — log and allow through with a warning.
        # Failing closed (blocking all traffic) is worse than failing open
        # when the API is temporarily down.
        logger.error(
            f"OpenAI Moderation API failed: {exc}. "
            "Allowing input through (fail-open)."
        )
        session_state[SESSION_TIMESTAMP_KEY] = time.time()
        return ModerationResult(
            passed=True,
            reason="",
            flagged_by="",
            moderation_used=False,
        )
