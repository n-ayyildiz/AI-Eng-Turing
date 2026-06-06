"""
SQLite persistence layer for Neurohypothesis v2.

Provides schema creation, connection management, and typed helper
functions for all five tables: users, sessions, hypotheses,
token_usage, and errors.

All queries use parameterised statements — no string concatenation.
Connection is opened per-operation and closed immediately to stay
safe for Streamlit's multi-threaded rerun model.

Public API:
    - init_db()                          — create tables if not exist
    - upsert_user(user_id)               — insert or update user row
    - create_session(session_id, user_id, topic)
    - close_session(session_id, n_hypotheses, completed)
    - save_hypothesis(session_id, hyp: dict) -> int (hyp_id)
    - update_hypothesis_feedback(hyp_id, rating, comment)
    - log_token_usage(session_id, node_name, model, in_tok, out_tok, cost)
    - log_error(session_id, node_name, error_type, message, recovered)
    - get_session_token_totals(session_id) -> dict
    - export_tables_to_csv(export_dir) -> list[Path]
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

import config

# =============================================================================
# Connection helper
# =============================================================================

def _connect() -> sqlite3.Connection:
    """
    Open and return a SQLite connection with row_factory set so rows
    behave like dicts. Foreign-key enforcement is enabled per connection.
    """
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# =============================================================================
# Schema creation
# =============================================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    first_seen TIMESTAMP NOT NULL,
    last_seen  TIMESTAMP NOT NULL,
    n_sessions INTEGER   NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id         TEXT    PRIMARY KEY,
    user_id            TEXT    NOT NULL REFERENCES users(user_id),
    started_at         TIMESTAMP NOT NULL,
    ended_at           TIMESTAMP,
    topic              TEXT,
    n_hypotheses_generated INTEGER NOT NULL DEFAULT 0,
    completed          BOOLEAN NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS hypotheses (
    hyp_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id                TEXT    NOT NULL REFERENCES sessions(session_id),
    hyp_index                 INTEGER NOT NULL,          -- 0, 1, 2
    text                      TEXT    NOT NULL,
    primary_category          TEXT,
    complementary_categories  TEXT,                      -- JSON list
    originality_score         REAL,
    originality_grade         TEXT,
    plausibility_avg          REAL,
    plausibility_breakdown    TEXT,                      -- JSON dict (6 dims)
    quality_gate_attempts     INTEGER NOT NULL DEFAULT 1,
    user_rating               INTEGER,                   -- 1-5 or NULL
    user_comment              TEXT
);

CREATE TABLE IF NOT EXISTS token_usage (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT      NOT NULL REFERENCES sessions(session_id),
    node_name    TEXT      NOT NULL,
    model        TEXT      NOT NULL,
    input_tokens INTEGER   NOT NULL DEFAULT 0,
    output_tokens INTEGER  NOT NULL DEFAULT 0,
    cost_usd     REAL      NOT NULL DEFAULT 0.0,
    timestamp    TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    err_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT      NOT NULL REFERENCES sessions(session_id),
    node_name    TEXT      NOT NULL,
    error_type   TEXT      NOT NULL,
    error_message TEXT     NOT NULL,
    recovered    BOOLEAN   NOT NULL DEFAULT 0,
    timestamp    TIMESTAMP NOT NULL
);
"""


def init_db() -> None:
    """
    Create all tables if they do not exist.

    Safe to call on every app startup — uses CREATE TABLE IF NOT EXISTS,
    so no data is lost on re-runs.
    """
    config.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.info(f"SQLite DB ready at {config.SQLITE_PATH}")


# =============================================================================
# Users
# =============================================================================

def upsert_user(user_id: str) -> None:
    """
    Insert a new user row or update last_seen + n_sessions for an
    existing user.

    Called once per session start in src/memory.py.
    """
    now = _now()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT user_id, n_sessions FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE users SET last_seen = ?, n_sessions = ? WHERE user_id = ?",
                (now, existing["n_sessions"] + 1, user_id),
            )
            logger.debug(f"User {user_id[:8]}… — session #{existing['n_sessions'] + 1}")
        else:
            conn.execute(
                "INSERT INTO users (user_id, first_seen, last_seen, n_sessions) "
                "VALUES (?, ?, ?, 1)",
                (user_id, now, now),
            )
            logger.info(f"New user registered: {user_id[:8]}…")
        conn.commit()


# =============================================================================
# Sessions
# =============================================================================

def create_session(session_id: str, user_id: str, topic: str) -> None:
    """
    Insert a new session row with status completed=False.

    Called at the start of each agent run.
    """
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, user_id, started_at, topic, n_hypotheses_generated, completed) "
            "VALUES (?, ?, ?, ?, 0, 0)",
            (session_id, user_id, _now(), topic),
        )
        conn.commit()
    logger.info(f"Session created: {session_id[:8]}… | topic: {topic[:60]}")


def close_session(session_id: str, n_hypotheses: int, completed: bool) -> None:
    """
    Mark a session as ended and record the final hypothesis count.

    Called from N20 (persist_session).
    """
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ?, n_hypotheses_generated = ?, completed = ? "
            "WHERE session_id = ?",
            (_now(), n_hypotheses, int(completed), session_id),
        )
        conn.commit()
    logger.info(f"Session closed: {session_id[:8]}… | hypotheses={n_hypotheses} | completed={completed}")


# =============================================================================
# Hypotheses
# =============================================================================

def save_hypothesis(session_id: str, hyp: dict[str, Any]) -> int:
    """
    Insert one hypothesis row and return its auto-generated hyp_id.

    Args:
        session_id: the parent session.
        hyp: a dict matching the Hypothesis TypedDict from agent_state.py.

    Returns:
        The SQLite rowid of the inserted hypothesis (used for feedback updates).
    """
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO hypotheses ("
            "  session_id, hyp_index, text, primary_category, "
            "  complementary_categories, originality_score, originality_grade, "
            "  plausibility_avg, plausibility_breakdown, quality_gate_attempts"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                hyp.get("index", 0),
                hyp.get("text", ""),
                hyp.get("primary_category"),
                json.dumps(hyp.get("complementary_categories", [])),
                hyp.get("originality_score"),
                hyp.get("originality_grade"),
                hyp.get("plausibility_avg"),
                json.dumps(hyp.get("plausibility_scores", {})),
                hyp.get("quality_gate_passes", 1),
            ),
        )
        conn.commit()
        hyp_id: int = cur.lastrowid  # type: ignore[assignment]
    logger.debug(f"Hypothesis saved: hyp_id={hyp_id} idx={hyp.get('index')} session={session_id[:8]}…")
    return hyp_id


def update_hypothesis_feedback(hyp_id: int, rating: int, comment: str | None) -> None:
    """
    Write user rating (1-5) and optional comment to an existing hypothesis row.

    Called from N18 (collect_feedback) after the user submits a rating.
    """
    with _connect() as conn:
        conn.execute(
            "UPDATE hypotheses SET user_rating = ?, user_comment = ? WHERE hyp_id = ?",
            (rating, comment, hyp_id),
        )
        conn.commit()
    logger.info(f"Feedback saved: hyp_id={hyp_id} rating={rating}")


# =============================================================================
# Token usage
# =============================================================================

def log_token_usage(
    session_id: str,
    node_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    """
    Append one token-usage row for a completed LLM call.

    Called from src/cost_tracking.py after every API call. All values
    are non-negative; cost_usd is computed from config.PRICING.
    """
    with _connect() as conn:
        conn.execute(
            "INSERT INTO token_usage "
            "(session_id, node_name, model, input_tokens, output_tokens, cost_usd, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, node_name, model, input_tokens, output_tokens, cost_usd, _now()),
        )
        conn.commit()


def get_session_token_totals(session_id: str) -> dict[str, Any]:
    """
    Return cumulative token counts and cost for a session.

    Used by the Streamlit sidebar live meter to update the running total
    without a full CSV export.

    Returns:
        Dict with keys: input_tokens, output_tokens, cost_usd.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT SUM(input_tokens) as i, SUM(output_tokens) as o, SUM(cost_usd) as c "
            "FROM token_usage WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    return {
        "input_tokens":  int(row["i"] or 0),
        "output_tokens": int(row["o"] or 0),
        "cost_usd":      float(row["c"] or 0.0),
    }


# =============================================================================
# Errors
# =============================================================================

def log_error(
    session_id: str,
    node_name: str,
    error_type: str,
    error_message: str,
    recovered: bool,
) -> None:
    """
    Record a degraded-operation error to the errors table.

    Args:
        session_id: the active session.
        node_name:  which LangGraph node encountered the error.
        error_type: exception class name or short label (e.g. "PubMedTimeout").
        error_message: str(exception) or a concise description.
        recovered:  True if the agent continued successfully after fallback.
    """
    with _connect() as conn:
        conn.execute(
            "INSERT INTO errors "
            "(session_id, node_name, error_type, error_message, recovered, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, node_name, error_type, error_message, int(recovered), _now()),
        )
        conn.commit()
    logger.warning(
        f"Error logged [{node_name}] {error_type}: {error_message[:120]} "
        f"({'recovered' if recovered else 'NOT recovered'})"
    )


# =============================================================================
# CSV export (developer panel)
# =============================================================================

def export_tables_to_csv(export_dir: Path | None = None) -> list[Path]:
    """
    Dump all five tables to CSV files in export_dir.

    Called from the developer panel "Export tables" button in app.py.

    Args:
        export_dir: destination directory (defaults to config.EXPORTS_DIR).

    Returns:
        List of Path objects for the written files, so the UI can offer
        download links.
    """
    out_dir = export_dir or config.EXPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = ["users", "sessions", "hypotheses", "token_usage", "errors"]
    written: list[Path] = []

    with _connect() as conn:
        for table in tables:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608 — table names are hardcoded, not user-supplied
            if not rows:
                logger.debug(f"export_tables_to_csv: table '{table}' is empty, skipping")
                continue

            path = out_dir / f"{table}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows([dict(row) for row in rows])

            written.append(path)
            logger.info(f"Exported {len(rows)} rows → {path}")

    return written


# =============================================================================
# Internal helpers
# =============================================================================

def _now() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(UTC).isoformat()
