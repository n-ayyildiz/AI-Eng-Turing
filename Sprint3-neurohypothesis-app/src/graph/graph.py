"""
LangGraph StateGraph definition for Neurohypothesis v2.1.

Wires nodes for three retrieval paths:

    Path A — local_only:    PDF ingest → N4 sub-pipeline → hypothesis loop
    Path B — pubmed_only:   pick primary → per-category retrieve → embed
                            → hypothesis loop
    Path C — combined:      run the PubMed per-category branch, then the shared
                            hypothesis loop, with uploaded-PDF evidence merged
                            into N9 alongside the PubMed evidence (Path C light —
                            no separate integration step)

Decision points:
    A   _decision_a()      — after N3:  local_only / pubmed_only / combined
    A2  _after_local()     — after N4c: local_only → straight to N8;
                                        combined → continue to PubMed branch
    D   _decision_d()      — after N16: pass → N17; fail → back to N13
    E   _decision_e()      — after N18: continue & under cap → N8;
                                        otherwise → N19

The N17 INTERRUPT is implemented with langgraph.types.interrupt() inside
the node function.  The graph compiles with MemorySaver checkpointer so
state persists across the interrupt.

Public API:
    build_graph()            -> CompiledStateGraph  (also writes topology to disk)
    export_graph_topology()  -> writes Mermaid .md + .png to exports/
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger
from src.agent_state import AgentState
from src.graph.nodes import (
    n1_validate_input,
    n2_parse_topic,
    n3_route_sources,
    n4a_ingest_pdfs,
    n4b_retrieve_local,
    n4c_extract_metadata_local,
    n5_embed_category_papers,
    n5_order_categories,
    n5_per_category_retrieve,
    n5_pick_primary,
    n8_select_category_for_hypothesis,
    n9_retrieve_evidence,
    n10_summarize_past,
    n11_summarize_future,
    n12_compute_gap,
    n13_generate_hypothesis,
    n14_score_originality,
    n15_judge_plausibility,
    n16_quality_gate,
    n17_present_to_user,
    n18_collect_feedback,
    n19_export_results,
    n20_persist_session,
    n_a_generate,
)

import config

# =============================================================================
# Conditional edge functions (routing logic only; no side effects)
# =============================================================================

def _after_n1(state: AgentState) -> str:
    """Did input validation pass?"""
    if not state.get("validation_passed", False):
        return "end_error"
    return "n2_parse_topic"


def _decision_d(state: AgentState) -> str:
    """Decision D — quality gate.

    PASS (or best-of-3) → N17 present to user
    FAIL                → N13 regenerate
    """
    if state.get("quality_gate_passed", False):
        return "n17_present_to_user"
    return "n13_generate_hypothesis"


def _decision_a(state: AgentState) -> str:
    """Decision A: which path to run?

    local_only  → PDF branch only (N4a..c), then hypothesis loop.
    pubmed_only → PubMed branch only (pick → order → retrieve → embed),
                  then hypothesis loop.
    combined    → run BOTH (we route to N4a first; after N4c continues to
                  the PubMed branch, then after embed continues to
                  hypothesis loop).
    """
    choice = state.get("path_choice", "pubmed_only")
    if choice in ("local_only", "combined"):
        return "n4a_ingest_pdfs"
    return "n5_pick_primary"


def _after_local(state: AgentState) -> str:
    """After N4c: skip PubMed (local_only) or continue (combined)?"""
    if state.get("path_choice") == "local_only":
        # Path A: no category structure — enter the shared gap pipeline at N9.
        # N9's local branch builds evidence from PDF metadata + RAG chunks.
        return "n9_retrieve_evidence"
    # combined: now run the PubMed branch.
    return "n5_pick_primary"


def _decision_e(state: AgentState) -> str:
    """Decision E — HITL continue/stop.

    For all three paths:
        continue + under cap → next hypothesis
        end-of-loop (cap hit or user stopped) → N19 export

    Path C (combined) runs as Path B's per-category loop with PDF evidence
    merged in N9; there is no separate integration step, so it ends at export
    exactly like Path B.
    """
    decision  = state.get("user_decision", "stop")
    next_idx  = state.get("current_hypothesis_index", 0)
    path = state.get("path_choice")

    # Path A: incremental loop — each 'continue' generates the NEXT hypothesis
    # (anchored on the next gap pair), up to PATH_A_MAX_HYPOTHESES.
    if path == "local_only":
        if decision == "continue" and next_idx < config.PATH_A_MAX_HYPOTHESES:
            return "n_a_generate"
        return "n19_export_results"

    if decision == "continue" and next_idx < config.MAX_HYPOTHESES:
        return "n8_select_category_for_hypothesis"

    return "n19_export_results"


# =============================================================================
# Graph construction
# =============================================================================

def build_graph():
    """
    Construct, wire, and compile the Neurohypothesis v2.1 StateGraph.

    Path C (combined) note: there is a SINGLE hypothesis loop shared by all
    paths.  Combined runs the PubMed per-category branch (N5*) and then the
    loop (N8–N16); N9 merges uploaded-PDF evidence into the same gap analysis
    as the PubMed evidence.  This is "Path C light" — no dual-branch loop and
    no end-of-loop integration node.  A true 6+6 dual-branch design can be
    added later if strict separation is ever required.
    """
    g = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    g.add_node("n1_validate_input",                    n1_validate_input)
    g.add_node("n2_parse_topic",                       n2_parse_topic)
    g.add_node("n3_route_sources",                     n3_route_sources)

    # Local-PDF branch
    g.add_node("n4a_ingest_pdfs",                      n4a_ingest_pdfs)
    g.add_node("n4b_retrieve_local",                   n4b_retrieve_local)
    g.add_node("n4c_extract_metadata_local",           n4c_extract_metadata_local)

    # PubMed per-category branch (v2.1 new)
    g.add_node("n5_pick_primary",                      n5_pick_primary)
    g.add_node("n5_order_categories",                  n5_order_categories)
    g.add_node("n5_per_category_retrieve",             n5_per_category_retrieve)
    g.add_node("n5_embed_category_papers",             n5_embed_category_papers)

    # Hypothesis loop
    g.add_node("n8_select_category_for_hypothesis",    n8_select_category_for_hypothesis)
    g.add_node("n9_retrieve_evidence",                 n9_retrieve_evidence)
    g.add_node("n10_summarize_past",                   n10_summarize_past)
    g.add_node("n11_summarize_future",                 n11_summarize_future)
    g.add_node("n12_compute_gap",                      n12_compute_gap)
    g.add_node("n13_generate_hypothesis",              n13_generate_hypothesis)
    g.add_node("n14_score_originality",                n14_score_originality)
    g.add_node("n15_judge_plausibility",               n15_judge_plausibility)
    g.add_node("n16_quality_gate",                     n16_quality_gate)

    # HITL + output
    g.add_node("n17_present_to_user",                  n17_present_to_user)
    g.add_node("n18_collect_feedback",                 n18_collect_feedback)
    g.add_node("n_a_generate",                         n_a_generate)
    g.add_node("n19_export_results",                   n19_export_results)
    g.add_node("n20_persist_session",                  n20_persist_session)

    # ── Entry & validation ────────────────────────────────────────────────
    g.add_edge(START, "n1_validate_input")
    g.add_conditional_edges(
        "n1_validate_input",
        _after_n1,
        {"end_error": END, "n2_parse_topic": "n2_parse_topic"},
    )

    g.add_edge("n2_parse_topic", "n3_route_sources")

    # ── Decision A: which path? ───────────────────────────────────────────
    g.add_conditional_edges(
        "n3_route_sources",
        _decision_a,
        {
            "n4a_ingest_pdfs":  "n4a_ingest_pdfs",
            "n5_pick_primary":  "n5_pick_primary",
        },
    )

    # ── Local-PDF branch ──────────────────────────────────────────────────
    g.add_edge("n4a_ingest_pdfs",          "n4b_retrieve_local")
    g.add_edge("n4b_retrieve_local",       "n4c_extract_metadata_local")
    g.add_conditional_edges(
        "n4c_extract_metadata_local",
        _after_local,
        {
            "n9_retrieve_evidence": "n9_retrieve_evidence",
            "n5_pick_primary":      "n5_pick_primary",
        },
    )

    # ── PubMed per-category branch ────────────────────────────────────────
    g.add_edge(     "n5_pick_primary",          "n5_order_categories")
    g.add_edge(     "n5_order_categories",      "n5_per_category_retrieve")
    g.add_edge(     "n5_per_category_retrieve", "n5_embed_category_papers")
    g.add_conditional_edges(
        "n5_embed_category_papers",
        lambda s: "end_error" if not s.get("validation_passed", True) else "n8_select_category_for_hypothesis",
        {"end_error": END, "n8_select_category_for_hypothesis": "n8_select_category_for_hypothesis"},
    )

    # ── Hypothesis generation loop ─────────────────────────────────────────
    g.add_edge("n8_select_category_for_hypothesis", "n9_retrieve_evidence")
    g.add_edge("n9_retrieve_evidence",     "n10_summarize_past")
    g.add_edge("n10_summarize_past",       "n11_summarize_future")
    g.add_edge("n11_summarize_future",     "n12_compute_gap")
    # After the gap is computed: Path A → batch generate; B/C → per-slot N13.
    g.add_conditional_edges(
        "n12_compute_gap",
        lambda s: "n_a_generate" if s.get("path_choice") == "local_only"
                  else "n13_generate_hypothesis",
        {
            "n_a_generate":            "n_a_generate",
            "n13_generate_hypothesis": "n13_generate_hypothesis",
        },
    )
    g.add_edge("n_a_generate",             "n17_present_to_user")
    g.add_edge("n13_generate_hypothesis",  "n14_score_originality")
    g.add_edge("n14_score_originality",    "n15_judge_plausibility")
    g.add_edge("n15_judge_plausibility",   "n16_quality_gate")

    # ── Decision D: quality gate (regenerate or present) ──────────────────
    g.add_conditional_edges(
        "n16_quality_gate",
        _decision_d,
        {
            "n17_present_to_user":      "n17_present_to_user",
            "n13_generate_hypothesis":  "n13_generate_hypothesis",
        },
    )

    # ── HITL + Decision E ─────────────────────────────────────────────────
    g.add_edge("n17_present_to_user",      "n18_collect_feedback")
    g.add_conditional_edges(
        "n18_collect_feedback",
        _decision_e,
        {
            "n8_select_category_for_hypothesis": "n8_select_category_for_hypothesis",
            "n_a_generate":                      "n_a_generate",
            "n17_present_to_user":               "n17_present_to_user",
            "n19_export_results":                "n19_export_results",
        },
    )

    # ── Final persistence ─────────────────────────────────────────────────
    g.add_edge("n19_export_results",        "n20_persist_session")
    g.add_edge("n20_persist_session",       END)

    # Compile with MemorySaver (required for HITL interrupt)
    compiled = g.compile(checkpointer=MemorySaver())

    # Best-effort: write a topology diagram to disk so the user can see
    # the wiring.  Failure here is non-fatal — the graph itself works fine.
    try:
        export_graph_topology(compiled)
    except Exception as exc:
        logger.warning(f"[build_graph] topology export skipped: {exc}")

    return compiled


# =============================================================================
# Graph topology export — writes Mermaid MD (always) and PNG (best-effort)
# =============================================================================

def export_graph_topology(
    compiled_graph,
    output_dir: Path | None = None,
) -> dict[str, Path | None]:
    """
    Write a visual diagram of the compiled graph to disk.

    Produces two files in `output_dir` (defaults to config.EXPORTS_DIR):

        graph_topology.md   — Markdown with an embedded Mermaid block.
                              Always written.  GitHub, VS Code, and many
                              Markdown viewers render Mermaid natively.

        graph_topology.png  — PNG rendered via mermaid.ink (network call).
                              Best-effort: if offline or the service is
                              unreachable, this file is skipped and a
                              warning is logged.  The MD still works.

    Both files are overwritten on each call.  Cheap to regenerate.

    Returns:
        {"md": Path | None, "png": Path | None}
        Each entry is the file path on success, None on failure.
    """
    output_dir = output_dir or config.EXPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path  = output_dir / "graph_topology.md"
    png_path = output_dir / "graph_topology.png"
    result: dict[str, Path | None] = {"md": None, "png": None}

    # ── Mermaid MD: always works (no network) ─────────────────────────────
    try:
        graph_obj   = compiled_graph.get_graph()
        mermaid_src = graph_obj.draw_mermaid()
        md_path.write_text(
            f"# Neurohypothesis v2.1 — graph topology\n\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"This Mermaid diagram is auto-generated by `build_graph()` at "
            f"app startup.  GitHub / VS Code / many Markdown viewers render "
            f"it directly; otherwise paste the contents of the code block "
            f"below into https://mermaid.live to view interactively.\n\n"
            f"```mermaid\n{mermaid_src}\n```\n",
            encoding="utf-8",
        )
        result["md"] = md_path
        logger.info(f"[graph] Mermaid topology written to {md_path}")
    except Exception as exc:
        logger.warning(f"[graph] Mermaid MD export failed: {exc}")

    # ── PNG: best-effort (uses mermaid.ink API; needs internet) ───────────
    try:
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        result["png"] = png_path
        logger.info(f"[graph] PNG topology written to {png_path}")
    except Exception as exc:
        logger.warning(
            f"[graph] PNG export skipped (mermaid.ink unreachable?): {exc}"
        )

    return result
