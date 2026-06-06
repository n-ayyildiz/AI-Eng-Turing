"""
Supabase feedback logger for Neurohypothesis.

Sends full session data to Supabase on session completion — one row per
session containing summary fields plus JSONB columns for hypotheses,
token usage breakdown, and errors.

Environment variables:
    SUPABASE_URL         — project URL  (https://xxxx.supabase.co)
    SUPABASE_SERVICE_KEY — service_role key  (Project Settings → API)

Public API:
    log_session(session_id, user_id, topic, hypotheses,
                cost_usd, path_choice) -> bool
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime

from loguru import logger


def _read_sqlite_detail(session_id: str) -> dict:
    """
    Read per-session rows from the local SQLite DB for Supabase export.
    Returns dicts for hypotheses, token_usage, and errors.
    Falls back to empty lists on any failure.
    """
    try:
        import config

        conn = sqlite3.connect(config.SQLITE_PATH)
        conn.row_factory = sqlite3.Row

        hyps = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM hypotheses  WHERE session_id = ?", (session_id,)
            ).fetchall()
        ]
        tokens = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM token_usage WHERE session_id = ?", (session_id,)
            ).fetchall()
        ]
        errors = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM errors      WHERE session_id = ?", (session_id,)
            ).fetchall()
        ]
        conn.close()
        return {"hypotheses_detail": hyps, "token_usage_detail": tokens, "errors_detail": errors}
    except Exception as exc:
        logger.warning(f"SQLite detail read failed: {exc}")
        return {"hypotheses_detail": [], "token_usage_detail": [], "errors_detail": []}


def log_session(
    session_id: str,
    user_id: str,
    topic: str,
    hypotheses: list[dict],
    cost_usd: float,
    path_choice: str,
) -> bool:
    """
    Insert one full session row into Supabase.
    Returns True on success, False on any failure (fail-open).
    """
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

    if not url or not key:
        logger.debug("Supabase not configured — skipping session log")
        return False

    try:
        from supabase import create_client

        client = create_client(url, key)

        # Full detail from SQLite (has correct ratings — N18 writes there)
        detail = _read_sqlite_detail(session_id)

        # Build a lookup: hyp_index → SQLite row (has user_rating/user_comment)
        sqlite_by_idx = {
            row.get("hyp_index", i): row for i, row in enumerate(detail["hypotheses_detail"])
        }

        ratings = [
            sqlite_by_idx.get(h.get("index", i), {}).get("user_rating")
            for i, h in enumerate(hypotheses)
            if sqlite_by_idx.get(h.get("index", i), {}).get("user_rating") is not None
        ]
        comments = [
            sqlite_by_idx.get(h.get("index", i), {}).get("user_comment") or ""
            for i, h in enumerate(hypotheses)
        ]
        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

        # Aggregate token totals from SQLite token_usage table
        total_input_tokens = sum(
            row.get("input_tokens", 0) or 0 for row in detail["token_usage_detail"]
        )
        total_output_tokens = sum(
            row.get("output_tokens", 0) or 0 for row in detail["token_usage_detail"]
        )

        # Hypothesis summary — merge state data (scores, text) with SQLite ratings
        hyp_summary = [
            {
                "index": h.get("index"),
                "text": h.get("text", "")[:600],
                "primary_category": h.get("primary_category"),
                "originality_score": h.get("originality_score"),
                "originality_grade": h.get("originality_grade"),
                "plausibility_avg": h.get("plausibility_avg"),
                "plausibility_scores": h.get("plausibility_scores"),
                "improvement_tips": h.get("improvement_tips"),
                "pubmed_check_grade": h.get("pubmed_check_grade"),
                "gap_score": h.get("gap_score"),
                # Ratings from SQLite (correct) not from state (always None)
                "user_rating": sqlite_by_idx.get(h.get("index", i), {}).get("user_rating"),
                "user_comment": sqlite_by_idx.get(h.get("index", i), {}).get("user_comment"),
                "contradictory_evidence": {
                    k: v
                    for k, v in (h.get("contradictory_evidence") or {}).items()
                    if k != "papers"
                },
            }
            for i, h in enumerate(hypotheses)
        ]

        client.table("neurohypothesis_sessions").insert(
            {
                "session_id": session_id,
                "user_id": user_id,
                "topic": topic[:500],
                "n_hypotheses": len(hypotheses),
                "ratings": ratings,
                "comments": comments,
                "avg_rating": avg_rating,
                "cost_usd": round(cost_usd, 6),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "path_choice": path_choice,
                "completed_at": datetime.now(UTC).isoformat(),
                # JSONB columns
                "hypotheses_summary": hyp_summary,
                "hypotheses_detail": detail["hypotheses_detail"],
                "token_usage_detail": detail["token_usage_detail"],
                "errors_detail": detail["errors_detail"],
            }
        ).execute()

        logger.info(
            f"Session logged to Supabase: {session_id[:8]}… "
            f"({len(hypotheses)} hyps, ${cost_usd:.4f})"
        )
        return True

    except ImportError:
        logger.error("Supabase package not installed. Run: pip install supabase")
        return False
    except Exception as exc:
        logger.error(f"Supabase log failed: {exc}")
        return False
