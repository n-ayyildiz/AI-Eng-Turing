"""
LangGraph state schema for Neurohypothesis.

Defines three TypedDicts that constitute the agent state:
    - Paper        — one retrieved or uploaded paper
    - Hypothesis   — one generated + scored + rated hypothesis
    - AgentState   — the full graph state passed between nodes

Design rules (unchanged from v2):
    - Each node reads only the slots it needs and writes only the slots
      it owns.  Partial dicts are returned and LangGraph merges them.
    - Lists are declared with Annotated[list, operator.add] so multiple
      nodes can safely append without overwriting each other's work.
    - Dicts use a custom merge_dicts reducer so sub-keys are updated
      without overwriting sibling keys set by other nodes.
    - Optional fields carry default None; no node should KeyError on read.

Public API:
    - Paper            TypedDict
    - Hypothesis       TypedDict
    - AgentState       TypedDict
    - merge_dicts      reducer helper (used by Annotated dict fields)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

try:
    from typing import TypedDict
except ImportError:                       # Python 3.11 compat shim (not needed for 3.12+)
    from typing_extensions import TypedDict


# =============================================================================
# Custom reducer for dict fields
# =============================================================================

def merge_dicts(existing: dict, update: dict) -> dict:
    """
    Merge two dicts by updating at the key level, not replacing the whole dict.

    Used as the LangGraph reducer for dict-typed state slots so that
    node A writing {"cat1": [...]} and node B writing {"cat2": [...]}
    produce {"cat1": [...], "cat2": [...]} rather than the second write
    overwriting the first.
    """
    if existing is None:
        return update or {}
    if update is None:
        return existing
    return {**existing, **update}


# =============================================================================
# Paper
# =============================================================================

class Paper(TypedDict, total=False):
    """
    Represents one paper from either a local PDF upload or a PubMed search.

    In v2.1, the `category` field is set by which per-category PubMed search
    retrieved the paper — not by a categorizer module.  `category_source`
    is no longer used (the deterministic categorizer is removed).
    """
    # ── Identity ──────────────────────────────────────────────────────────
    pmid:             str | None
    source:           Literal["local", "pubmed"]

    # ── Bibliographic ─────────────────────────────────────────────────────
    title:            str
    abstract:         str
    full_text:        str | None             # only for local PDFs
    journal:          str
    year:             int
    authors:          list[str]

    # ── PubMed metadata ───────────────────────────────────────────────────
    mesh_terms:       list[str]
    publication_type: list[str]
    url:              str | None             # canonical PubMed URL (not shown in v2.1 UI)

    # ── Classification (v2.1: set by which search retrieved the paper) ────
    category:         str | None             # one of the 6 category strings

    # ── Per-paper relevance to query (v2.1) ───────────────────────────────
    query_cosine:     float | None           # cosine(user_query, abstract) for ranking + gate

    # ── Extracted metadata (kept internally, hidden from UI) ──────────────
    metadata:         dict[str, Any] | None  # goal, methods, results, future (LLM-extracted)


# =============================================================================
# Hypothesis
# =============================================================================

class Hypothesis(TypedDict, total=False):
    """
    Represents one generated, scored, and user-rated hypothesis.

    Populated incrementally across nodes:
        N8  → index, primary_category, complementary_categories
        N13 → text
        N14 → originality_score, originality_grade
        N15 → plausibility_scores, plausibility_avg
        N16 → quality_gate_passes
        N17 → (presented to user)
        N18 → user_rating, user_comment, db_hyp_id
    """
    index:                    int                         # 0..MAX_HYPOTHESES-1
    text:                     str
    statement:                str | None               # canonical hypothesis text used by N13/N14 + Path A

    # ── Category combination ──────────────────────────────────────────────
    primary_category:         str
    complementary_categories: list[str]

    # ── Originality (N14) ─────────────────────────────────────────────────
    originality_score:        float
    originality_grade:        str    # very/moderate/less or label

    # ── Plausibility (N15) ────────────────────────────────────────────────
    plausibility_scores:      dict[str, float]
    plausibility_avg:         float
    plausibility_verdict:     str

    # ── PubMed-check evaluator (v1 freshness check, kept) ─────────────────
    pubmed_check_score:       float | None
    pubmed_check_grade:       str | None
    pubmed_check_n_found:     int | None

    # ── Quality gate metadata (N16) ───────────────────────────────────────
    quality_gate_passes:      int
    low_confidence:           bool

    # ── Evidence ──────────────────────────────────────────────────────────
    sources_used:             list[str]    # cited paper IDs / PMIDs (subset of retrieved)
    sources_retrieved:        list[str]    # full retrieved set (all 10 or 20, by PMID)
    past_summary:             str
    future_summary:           str
    gap_score:                float
    suggested_approach:       list[str]

    # ── User feedback (N18) ───────────────────────────────────────────────
    user_rating:              int | None
    user_comment:             str | None
    db_hyp_id:                int | None



# =============================================================================
# AgentState  (v2.1 — old PubMed primary/alt slots removed; per-category added)
# =============================================================================

class AgentState(TypedDict, total=False):
    """
    Full typed state for the Neurohypothesis v2.1 LangGraph agent.

    Reducer annotations:
        list fields → Annotated[list, operator.add]  (safe append from any node)
        dict fields → Annotated[dict, merge_dicts]   (key-level merge, not replace)
        scalar fields → last-writer-wins (LangGraph default)
    """

    # ── User input ────────────────────────────────────────────────────────
    user_id:            str
    session_id:         str
    topic:              str
    parsed_topic:       dict[str, str]      # {primary_method, primary_domain, focus}
    has_uploaded_pdfs:  bool
    pdf_paths:          list[str]

    # ── Path routing (v2.1, new) ──────────────────────────────────────────
    # Set by the UI before graph runs.  Drives the three-branch routing.
    path_choice:        Literal["local_only", "pubmed_only", "combined"]

    # Legacy slot kept for backward-compat with code that still reads it.
    # In v2.1 it mirrors path_choice but with the v2 vocabulary ("both" == "combined").
    source_decision:    Literal["local_only", "pubmed_only", "both"]

    # ── Per-category retrieval (v2.1, new — Path B core) ──────────────────
    # The LLM-picked primary category for THIS query.
    primary_category:           str

    # [primary, comp_a, comp_b, comp_c, comp_d, comp_e] in alphabetical order
    # of the complementary 5.  H1 uses index 0; H2..H6 pair primary + index 1..5.
    ordered_categories:         list[str]

    # Per-category audit trail of reformulation attempts (shown in progress stream).
    # Shape: { category_name: [ {attempt:1, temperature:0.0, reformulation:str,
    #                            mesh_terms:[...], quality_score:float,
    #                            n_retrieved:int, n_relevant:int,
    #                            mean_cosine:float, passed:bool}, ...] }
    category_reformulations:    Annotated[dict[str, list[dict]], merge_dicts]

    # Final retained papers per category, after the 3-attempt retrieval loop.
    # Empty list (or missing key) means "no relevant papers found in this category".
    category_papers:            Annotated[dict[str, list], merge_dicts]

    # Per-category retrieval-quality stats (shown on hypothesis card).
    # Shape: { category_name: {mean_cosine: float, n_relevant: int,
    #                          n_retrieved: int, attempts: int,
    #                          low_relevance_badge: bool} }
    category_relevance:         Annotated[dict[str, dict], merge_dicts]

    # ── Hypothesis generation loop ────────────────────────────────────────
    # The canonical accumulating list shown to the user.
    # Filled by N16 (Paths B/C) or N_a (Path A).
    hypotheses:                 Annotated[list[Hypothesis], operator.add]


    current_hypothesis_index:   int                                         # 0..MAX-1
    quality_gate_attempts:      Annotated[dict[str, int], merge_dicts]      # str(idx) → attempts

    # ── Local PDF metadata (N4c) ──────────────────────────────────────────
    # Kept internally for gap analysis; not exposed in v2.1 UI.
    local_paper_metadata:       Annotated[list[dict], operator.add]

    # ── Routing flags ─────────────────────────────────────────────────────
    validation_passed:          bool
    quality_gate_passed:        bool
    quality_gate_is_best_of:    bool

    # ── HITL (Decision E — N17–N18) ───────────────────────────────────────
    user_decision:              Literal["continue", "stop"] | None
    user_rating:                int | None
    user_comment:               str | None

    # ── Observability ─────────────────────────────────────────────────────
    token_usage:                Annotated[dict[str, Any], merge_dicts]
    errors:                     Annotated[list[dict], operator.add]
    node_timings:               Annotated[dict[str, float], merge_dicts]

    # ── Transient slots between adjacent nodes (v2.1 fix May 11) ──────────
    # These were previously undeclared and relied on TypedDict total=False to
    # let LangGraph propagate them.  In langgraph 1.x undeclared keys are
    # filtered, so N9 → N10/N13/N16 state-flow broke and every hypothesis
    # rendered empty.  Declared explicitly below.  All scalar/list/dict
    # last-writer-wins (no merge reducer needed — each iteration overwrites).
    _evidence_past:        list[dict]
    _evidence_future:      list[dict]
    _evidence_pids:        list[str]
    _path_c_pdf_pids:      list[str]   # Path C: PDF ids front-loaded into the summary window (credited in sources_used)
    _past_summary:         str
    _future_summary:       str
    _gap_score:            float
    _current_hypothesis:   dict
    _gap_pairs_ordered:    list[dict]   # Path A: pre-ordered gap pairs, one per slot
    _current_primary_cat:           str
    _current_comp_cats:             list[str]
    _current_intended_comp_cats:    list[str]
    _originality_result:   dict
    _plausibility_result:  dict
    _gate_failure_reason:  str | None
