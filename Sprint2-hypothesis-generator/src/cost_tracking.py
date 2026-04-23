"""
Token usage and cost tracking.

Uses tiktoken to count tokens before each LLM call, logs input/output
token counts and cost estimates, and exposes running session totals
to the Streamlit sidebar.

All tracking is done via a singleton SessionCostTracker stored in
Streamlit's session_state. Each LLM or embedding call should be
logged by calling log_call() after the response is received.

Public API:
    - get_tracker() -> SessionCostTracker
    - count_tokens(text, model) -> int
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import tiktoken
import streamlit as st

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Token counting
# =============================================================================

_encoder_cache: dict[str, tiktoken.Encoding] = {}


def count_tokens(text: str, model: str = config.MAIN_LLM_MODEL) -> int:
    """
    Count the number of tokens in a text string for a given model.

    Uses tiktoken's encoding for the specified model. Falls back to
    cl100k_base (GPT-4 / GPT-4o family) if the model is not recognised.
    """
    if model not in _encoder_cache:
        try:
            _encoder_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _encoder_cache[model] = tiktoken.get_encoding("cl100k_base")

    return len(_encoder_cache[model].encode(text))


# =============================================================================
# Call log entry
# =============================================================================

@dataclass
class CallLogEntry:
    """One logged API call with token counts and estimated cost."""
    timestamp: str
    model: str
    call_type: str
    summary: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


# =============================================================================
# Session cost tracker
# =============================================================================

@dataclass
class SessionCostTracker:
    """
    Accumulates token usage and cost estimates across all API calls
    in a single Streamlit session.
    """
    calls: list[CallLogEntry] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def log_call(
        self,
        model: str,
        call_type: str,
        summary: str,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> CallLogEntry:
        """
        Log one API call and update running totals.

        Cost is estimated using the pricing table in config.py.
        Prices are per 1M tokens.
        """
        pricing = config.PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )

        entry = CallLogEntry(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            model=model,
            call_type=call_type,
            summary=summary,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )

        self.calls.append(entry)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        logger.info(
            f"API call logged: {summary} | {model} | "
            f"in={input_tokens} out={output_tokens} | ${cost:.6f}"
        )

        return entry


def get_tracker() -> SessionCostTracker:
    """
    Return the session-scoped cost tracker, creating it if it does not
    exist yet.
    """
    if "cost_tracker" not in st.session_state:
        st.session_state.cost_tracker = SessionCostTracker()
    return st.session_state.cost_tracker
