"""
Token usage and cost tracking for Neurohypothesis v2.

Extended from v1: every LLM call is now also written to the SQLite
token_usage table so the developer can export and analyse costs offline,
and the per-node breakdown is visible in the dev panel.

Design:
    - SessionCostTracker is a dataclass stored in st.session_state.
    - Every LLM/embedding call passes node_name so costs are attributed
      to the specific graph node that made them.
    - log_call() updates the in-memory tracker AND writes a row to SQLite.
    - The sidebar live meter reads from the in-memory tracker (no DB round-
      trip per Streamlit rerun).

Public API:
    - get_tracker() -> SessionCostTracker
    - count_tokens(text, model) -> int
    - SessionCostTracker.log_call(node_name, model, call_type, summary,
                                   input_tokens, output_tokens)
    - SessionCostTracker.set_session(user_id, session_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import streamlit as st
import tiktoken
from loguru import logger

import config

# =============================================================================
# Token counting (tiktoken, cached by model name)
# =============================================================================

_encoder_cache: dict[str, tiktoken.Encoding] = {}


def count_tokens(text: str, model: str = config.MAIN_LLM_MODEL) -> int:
    """
    Count the number of tokens in a text string for a given model.

    Uses tiktoken for the specified model; falls back to cl100k_base
    (GPT-4 family) if the model string is not recognised by tiktoken.

    Args:
        text:  the string to count tokens in.
        model: the model name used to select the encoding.

    Returns:
        Token count as int.
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
    """One logged API call — held in SessionCostTracker.calls."""
    timestamp:         str
    node_name:         str      # which LangGraph node made this call
    model:             str
    call_type:         str      # "llm" | "embedding" | "moderation"
    summary:           str      # short human-readable description
    input_tokens:      int
    output_tokens:     int
    estimated_cost_usd: float


# =============================================================================
# Session cost tracker
# =============================================================================

@dataclass
class SessionCostTracker:
    """
    Accumulates token usage and cost estimates across all API calls
    in one LangGraph session.

    Stored in st.session_state so it survives Streamlit reruns within
    the same browser session.  Writes to SQLite on every log_call().
    """
    calls:               list[CallLogEntry] = field(default_factory=list)
    total_input_tokens:  int   = 0
    total_output_tokens: int   = 0
    total_cost_usd:      float = 0.0

    # Set by set_session() once the LangGraph run starts.
    session_id: str | None = None
    user_id:    str | None = None

    def set_session(self, user_id: str, session_id: str) -> None:
        """Attach user_id and session_id for SQLite writes."""
        self.user_id    = user_id
        self.session_id = session_id

    def log_call(
        self,
        node_name:     str,
        model:         str,
        call_type:     str,
        summary:       str,
        input_tokens:  int,
        output_tokens: int = 0,
    ) -> CallLogEntry:
        """
        Log one API call, update running totals, and persist to SQLite.

        Args:
            node_name:     the LangGraph node that made this call
                           (e.g. "N2_parse_topic", "N5b_pubmed_search").
            model:         model string — must match a key in config.PRICING
                           for cost to be non-zero.
            call_type:     "llm" | "embedding" | "moderation".
            summary:       short description for the dev panel log.
            input_tokens:  counted before the call via count_tokens().
            output_tokens: counted after the call (0 for embeddings).

        Returns:
            The CallLogEntry that was stored.
        """
        pricing = config.PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (
            input_tokens  * pricing["input"]  / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )

        entry = CallLogEntry(
            timestamp=_now(),
            node_name=node_name,
            model=model,
            call_type=call_type,
            summary=summary,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )

        self.calls.append(entry)
        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd      += cost

        logger.info(
            f"[{node_name}] {call_type} | {model} | "
            f"in={input_tokens} out={output_tokens} | ${cost:.6f} | {summary}"
        )

        # Persist to SQLite if session is active
        if self.session_id:
            try:
                from src.db import log_token_usage  # local import avoids circular dep
                log_token_usage(
                    session_id=self.session_id,
                    node_name=node_name,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )
            except Exception as exc:
                # Token logging must never crash the agent.
                logger.warning(f"SQLite token_usage write failed: {exc}")

        return entry

    def node_breakdown(self) -> dict[str, dict[str, float]]:
        """
        Aggregate token counts and cost per node_name.

        Returns:
            Dict mapping node_name → {input_tokens, output_tokens, cost_usd}.
            Used by the dev panel per-node timing/cost table.
        """
        breakdown: dict[str, dict[str, float]] = {}
        for entry in self.calls:
            row = breakdown.setdefault(
                entry.node_name,
                {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            row["input_tokens"]  += entry.input_tokens
            row["output_tokens"] += entry.output_tokens
            row["cost_usd"]      += entry.estimated_cost_usd
        return breakdown


# =============================================================================
# Session-scoped singleton
# =============================================================================

def get_tracker() -> SessionCostTracker:
    """
    Return the session-scoped cost tracker from st.session_state,
    creating a fresh one if it doesn't exist yet.

    Call this at the start of every LangGraph node that makes LLM calls.
    """
    if "cost_tracker" not in st.session_state:
        st.session_state.cost_tracker = SessionCostTracker()
    return st.session_state.cost_tracker


# =============================================================================
# Internal helpers
# =============================================================================

def _now() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")
