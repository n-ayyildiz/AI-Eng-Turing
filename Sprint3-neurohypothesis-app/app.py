"""
Neurohypothesis — Streamlit UI.

Phases:
    input    — collect topic + optional PDFs + path choice
    running  — graph streams until first interrupt (N17)
    feedback — show hypothesis card + rating + continue/stop
    done     — read-only stream of all hypotheses + PDF export
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from langgraph.types import Command
from loguru import logger
from src.cost_tracking import get_tracker
from src.engine.exports import render_pdf_bytes
from src.feedback import log_session

import config

# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="Neurohypothesis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# One-time DB + logger init
# =============================================================================

def _init_db() -> None:
    from src.db import init_db
    if "_db_inited" not in st.session_state:
        init_db()
        st.session_state["_db_inited"] = True


def _init_logging() -> None:
    if "_log_inited" in st.session_state:
        return
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(config.LOG_FILE, level="DEBUG", rotation="10 MB", retention=3)
    st.session_state["_log_inited"] = True


def _get_graph():
    """Build and cache the graph once per app lifetime."""
    if "_graph" not in st.session_state:
        logger.info("Building LangGraph StateGraph…")
        from src.graph.graph import build_graph
        st.session_state["_graph"] = build_graph()
    return st.session_state["_graph"]


# =============================================================================
# CSS  (v2.1: cleaner card; closed-by-default expanders; no badge-low ribbon
#            since low_confidence shown inline)
# =============================================================================

st.markdown("""
<style>
/* ──────────────────────────────────────────────────────────────────
   ────────────────────────────────────────────────────────────────── */

/* === Purple palette (matches button "fitting but lighter for scores") ===
   --primary:        #5e3a87 (deep purple — buttons, hyp-card header)
   --primary-hover:  #7c5fa6 (medium purple — button hover)
   --score:          #8a6cb4 (lighter purple — score badges, reformulation)
   --section:        #6d4794 (medium-dark purple — expander section titles)
*/

/* Main container — extra bottom padding for the sticky footer */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px;
}

/* Headings */
h1 {
    font-size: 1.95rem !important;
    margin: 0 0 0.6rem !important;
    color: #1a4f8a !important;
    letter-spacing: 0.01em !important;
    line-height: 1.2 !important;
}
h2 { font-size: 1.2rem !important; margin: 0.3rem 0 0.3rem !important; }
h3 { font-size: 1.05rem !important; margin: 0.3rem 0 0.2rem !important; }

/* Body text */
p, li { line-height: 1.4 !important; font-size: 0.92rem; }
.stMarkdown p { margin-bottom: 0.35rem; }

/* Sidebar density */
[data-testid="stSidebar"] .block-container {
    padding: 0.3rem 0.8rem 0.8rem !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stSidebar"] h2 { font-size: 0.95rem !important; margin-top: 0 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] .stCaption {
    font-size: 0.72rem !important;
    line-height: 1.3 !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-size: 1rem !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
}
[data-testid="stSidebar"] hr { margin: 0.4rem 0 !important; }

/* Expanders — tighter + purple section titles */
[data-testid="stExpander"] {
    margin: 0.15rem 0 0.15rem !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {
    padding: 0.25rem 0.5rem !important;
    font-size: 0.85rem !important;
    color: #6d4794 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] details > div {
    padding: 0.4rem 0.6rem !important;
}

/* Input widgets — tighter */
[data-testid="stTextArea"] textarea {
    font-size: 0.88rem !important;
    line-height: 1.35 !important;
}
[data-testid="stTextInput"] input {
    font-size: 0.88rem !important;
    padding: 0.35rem 0.5rem !important;
}

/* Radio buttons — tighter horizontal spacing */
.stRadio [role="radiogroup"] {
    gap: 0.6rem !important;
    padding: 0.1rem 0 !important;
}
.stRadio label,
.stRadio label p {
    font-size: 0.85rem !important;
    margin: 0 !important;
}

/* Buttons — compact */
.stButton button {
    font-size: 0.85rem !important;
    padding: 0.3rem 0.8rem !important;
    min-height: unset !important;
}

/* Primary button — purple */
button[kind="primary"] {
    background-color: #5e3a87 !important;
    border-color: #5e3a87 !important;
    color: white !important;
}
button[kind="primary"]:hover {
    background-color: #7c5fa6 !important;
    border-color: #7c5fa6 !important;
}

/* File uploader — compact */
[data-testid="stFileUploaderDropzone"] {
    min-height: unset !important;
    padding: 0.4rem 0.6rem !important;
}
[data-testid="stFileUploaderDropzone"] small {
    font-size: 0.7rem !important;
}

/* Hypothesis card — denser, purple header */
.hyp-card {
    border: 1px solid #e0d8ea;
    border-radius: 6px;
    padding: 0.7rem 1rem 0.4rem;
    margin-bottom: 0.6rem;
    background: #faf8fc;
}
.hyp-header {
    background: #1a4f8a;
    color: white;
    padding: 0.3rem 0.7rem;
    border-radius: 4px 4px 0 0;
    font-weight: 600;
    font-size: 0.95rem;
    margin: -0.7rem -1rem 0.5rem;
}

/* Score badges — light purple */
.score-badges {
    font-size: 0.82rem;
    color: #8a6cb4;
    margin: 0.3rem 0 0.15rem;
}
.summary-line {
    font-size: 0.75rem;
    color: #666666;
    margin: 0 !important;
}
.badge-low {
    color: #E9A23B;
    font-weight: 600;
    font-size: 0.82rem;
    margin: 0.2rem 0 !important;
}
.reformulation {
    color: #8a6cb4;
    font-size: 0.82rem;
    margin-left: 1.2rem;
    font-style: italic;
}

/* Divider — thinner spacing */
hr { margin: 0.5rem 0 !important; }

/* Status (spinner) box — tighter */
[data-testid="stStatus"] {
    margin: 0.3rem 0 !important;
}

/* Hide sidebar collapse arrows */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[kind="header"][aria-label*="sidebar" i],
button[kind="headerNoPadding"] {
    display: none !important;
}

/* Defensive: kill any stale file-uploader native <input> outside its
   intended block (addresses "clicking during running phase
   opens file picker" bug). */
input[type="file"]:not([data-testid]) {
    pointer-events: none !important;
}
[data-testid="stFileUploader"]:empty {
    display: none !important;
    pointer-events: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Sticky bottom disclaimer — needs to appear during ALL
   phases including the long-running "running" phase, where the main script
   blocks inside _stream_graph and never reaches the end-of-main() disclaimer.
   This is rendered EARLY in main() inside a fixed-position div, so it shows
   regardless of where the script is blocked. */
/* In-flow disclaimer: a normal block at the bottom of the main content, so it
   shares the .block-container box — its left edge lines up with the title and
   it flexes with the sidebar automatically.  (Not pinned during the long run.) */
.disclaimer-footer {
    border-top: 1px solid #e0d8ea;
    margin-top: 1.5rem;
    padding-top: 0.5rem;
    font-size: 0.72rem;
    color: #555;
    line-height: 1.3;
    text-align: left;
}
.disclaimer-footer .warn { color: #E9A23B; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session state initialisation
# =============================================================================

def _init_session() -> None:
    defaults = {
        "user_id":         str(uuid.uuid4()),
        "session_id":      str(uuid.uuid4()),
        "phase":           "input",
        "thread":          None,
        "pdf_tmp_paths":   [],
        "hypotheses":      [],
        "path_choice":     None,         # set in input phase
        "export_bytes":    None,
        "_reformulations": {},           # category → list[dict] for inline display
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =============================================================================
# Friendly node labels for progress log
# =============================================================================

_NODE_LABELS: dict[str, str] = {
    "n1_validate_input":             "🔒 Validating input",
    "n2_parse_topic":                "🔍 Parsing research topic",
    "n3_route_sources":              "🗺 Routing path",
    "n4a_ingest_pdfs":               "📄 Ingesting PDFs",
    "n4b_retrieve_local":            "🔎 Testing local retrieval",
    "n4c_extract_metadata_local":    "📝 Extracting local paper metadata",
    "n5_pick_primary":               "🎯 Picking primary category",
    "n5_order_categories":           "📋 Ordering categories",
    "n5_per_category_retrieve":      "🌐 Searching PubMed per category",
    "n5_embed_category_papers":      "📦 Embedding abstracts",
    "n8_select_category_for_hypothesis": "🎯 Selecting category pair",
    "n9_retrieve_evidence":          "🔎 Retrieving evidence",
    "n10_summarize_past":            "📖 Summarising past findings",
    "n11_summarize_future":          "🔭 Summarising future directions",
    "n12_compute_gap":               "📐 Computing literature gap",
    "n13_generate_hypothesis":       "💡 Generating hypothesis",
    "n14_score_originality":         "🎲 Scoring originality",
    "n15_judge_plausibility":        "⚖️ Judging plausibility",
    "n16_quality_gate":              "🚦 Quality gate",
    "n17_present_to_user":           "⏸ Presenting hypothesis (waiting for you…)",
    "n18_collect_feedback":          "💾 Saving your feedback",
    "n19_export_results":            "📄 Exporting PDF report",
    "n20_persist_session":           "🗄 Persisting session",
}


# =============================================================================
# APA citation helper (v2.1 — no volume/issue/pages/DOI, no link)
# =============================================================================

def _format_apa(paper: dict) -> str:
    """
    Format a paper as a simple APA-style line.

    Author, A., Author, B., & Author, C. (Year). Title. Journal.

    Falls back gracefully when fields are missing.
    """
    authors = paper.get("authors", []) or []
    if not authors:
        author_str = ""
    elif len(authors) == 1:
        author_str = authors[0]
    elif len(authors) <= 6:
        author_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
    else:
        author_str = ", ".join(authors[:6]) + ", et al."

    year    = str(paper.get("year") or "").strip()
    title   = str(paper.get("title") or "Untitled").rstrip(".")
    journal = str(paper.get("journal") or "").strip()

    # Assemble only the parts that exist — uploaded PDFs often lack
    # authors/year/journal, so avoid "Anonymous (n.d.). … . **.".
    if author_str and year:
        lead = f"{author_str} ({year}). "
    elif author_str:
        lead = f"{author_str}. "
    elif year:
        lead = f"({year}). "
    else:
        lead = ""
    tail = f" *{journal}*." if journal else ""
    return f"{lead}{title}.{tail}"


# =============================================================================
# Hypothesis card renderer  (v2.1 layout)
# =============================================================================

def _render_hypothesis_card(
    hyp:                dict,
    cat_relevance:      dict[str, dict] | None = None,
    cat_papers:         dict[str, list] | None = None,
    show_feedback_form: bool                   = False,
) -> dict | None:
    """
    Render one hypothesis with the v2.1 closed-by-default expander layout.

    cat_relevance and cat_papers are read-only context dicts the card uses
    to populate the per-category Sources sub-expander.  Pass None for
    Path A hypotheses (no category-level retrieval).
    """
    idx      = hyp.get("index", 0)
    label    = f"Hypothesis {idx + 1}"
    primary  = hyp.get("primary_category", "")
    comp          = hyp.get("complementary_categories", [])
    intended_comp = hyp.get("intended_complementary_categories", comp)

    is_local_only = st.session_state.get("path_choice") == "local_only"

    if is_local_only:
        # Path A has no category structure — show a clean header, no complement note.
        cat_text = ""
    elif comp:
        cat_text = primary + f", complemented by {', '.join(comp)}"
    elif idx > 0 and intended_comp and intended_comp != [primary]:
        cat_text = (
            f"{primary} "
            f"<span style='color:#e07b00;font-size:0.85em'>"
            f"(couldn't complement by {', '.join(intended_comp)})"
            f"</span>"
        )
    elif idx > 0:
        cat_text = (
            f"{primary} "
            f"<span style='color:#e07b00;font-size:0.85em'>"
            f"(couldn't complement by any other category)"
            f"</span>"
        )
    else:
        cat_text = primary

    # Header omits the "· category" segment entirely when there is no category.
    header_inner = f"{label} &nbsp;·&nbsp; {cat_text}" if cat_text else label

    feedback = None
    with st.container():
        st.markdown(f"""
<div class="hyp-card">
  <div class="hyp-header">{header_inner}</div>
""", unsafe_allow_html=True)

        if hyp.get("low_confidence"):
            st.markdown(
                '<p class="badge-low">⚠ Best-of-3 — quality gate not fully met '
                'after 3 attempts</p>',
                unsafe_allow_html=True,
            )

        # ── Hypothesis text ──────────────────────────────────────────────
        st.markdown(f"**{hyp.get('text', '')}**")

        # ── Path C: uploaded PDFs skipped for low query relevance ────────
        if idx == 0 and st.session_state.get("path_choice") == "combined":
            _dropped = [
                p for p in st.session_state.get("_local_paper_metadata", [])
                if float(p.get("query_cosine") or 0.0) < config.PATH_C_PDF_MIN_QUERY_COSINE
            ]
            if _dropped:
                _names = ", ".join(p.get("title", "?") for p in _dropped[:5])
                st.warning(
                    f"{len(_dropped)} uploaded PDF(s) were not used — low relevance "
                    f"to your query: {_names}.",
                    icon="⚠️",
                )

        # ── Complementary category unavailable — short notice ────────────
        if idx > 0 and not comp and not is_local_only:
            if intended_comp and intended_comp != [primary]:
                st.warning(
                    f"Generated from **{primary}** only — PubMed couldn't find "
                    f"relevant papers in **{', '.join(intended_comp)}**.",
                    icon="⚠️",
                )
            else:
                st.warning(
                    f"Generated from **{primary}** only — PubMed couldn't find "
                    f"relevant papers in any complementary category.",
                    icon="⚠️",
                )

        # ── Primary / complement paper availability warnings ──────────────
        primary_papers = len((cat_papers or {}).get(primary, []))
        comp_papers    = sum(len((cat_papers or {}).get(c, [])) for c in comp)

        if not is_local_only and primary_papers == 0 and comp_papers == 0:
            all_cats = f"**{primary}**" + (f" or **{', '.join(comp)}**" if comp else "")
            st.warning(
                f"PubMed couldn't find relevant papers in {all_cats} — "
                f"evidence was drawn from all other available categories as fallback.",
                icon="⚠️",
            )
        elif not is_local_only and primary_papers == 0 and comp_papers > 0:
            st.warning(
                f"PubMed couldn't find relevant papers in **{primary}** — "
                f"evidence was drawn from all available categories as fallback "
                f"(including **{', '.join(comp)}**).",
                icon="⚠️",
            )

        # ── Score badge row ─
        orig_score = hyp.get("originality_score")
        plaus_avg  = hyp.get("plausibility_avg")
        pmc_score  = hyp.get("pubmed_check_score")
        pmc_grade  = hyp.get("pubmed_check_grade")

        badges: list[str] = []
        if orig_score is not None:
            badges.append(f"Originality {orig_score:.2f} / 1 (vs your library)")
        if plaus_avg is not None:
            badges.append(f"Plausibility {plaus_avg:.1f} / 5 (LLM judge)")
        # PubMed novelty replaces the gap badge (gap lives in the Gap Analysis section).
        if pmc_score is not None:
            badges.append(f"Novelty vs PubMed {pmc_score:.2f} / 1 (25 yr)")
        elif pmc_grade:
            badges.append(f"Novelty vs PubMed: {pmc_grade}")
        if badges:
            st.markdown(
                '<p class="score-badges">' + '  ·  '.join(badges) + '</p>',
                unsafe_allow_html=True,
            )

        # ── Plausibility breakdown ─────────────────────
        # Shows the 6 per-dimension scores from the LLM judge and flags
        # any dimension < 3 so the user can spot bad hypotheses even when
        # the overall average is high (e.g. a logically inconsistent
        # design with avg=4.3 but testability=2).
        plaus_scores = hyp.get("plausibility_scores") or {}
        low_dims     = [d for d, s in plaus_scores.items() if s < 3]
        if low_dims:
            st.markdown(
                f'<p class="badge-low">'
                f'⚠ Weak dimensions (below 3 / 5): {", ".join(low_dims)}'
                f'</p>',
                unsafe_allow_html=True,
            )
        if plaus_scores:
            with st.expander("Plausibility breakdown — per dimension", expanded=False):
                for dim in (
                    "novelty", "testability", "mechanistic_coherence",
                    "citation_traceability", "conflict_awareness", "usefulness",
                ):
                    s = plaus_scores.get(dim)
                    if s is not None:
                        bar  = "★" * int(s) + "☆" * (5 - int(s))
                        flag = "  ⚠" if s < 3 else ""
                        st.markdown(
                            f"- **{dim.replace('_', ' ').title()}**: "
                            f"{bar}  ({int(s)} / 5){flag}"
                        )
                # Show improvement tips for weak dimensions instead of raw verdict
                tips = hyp.get("improvement_tips") or []
                if tips:
                    st.markdown("**💡 How to strengthen this hypothesis:**")
                    for tip in tips:
                        st.caption(f"• {tip}")

        # ── Contradictory evidence ────────────────────────────────────────
        contra = hyp.get("contradictory_evidence") or {}
        if contra:
            found  = contra.get("found", False)
            n_found = contra.get("n_found", 0)
            papers  = contra.get("papers", [])
            label   = (
                f"⚠️ Contradictory evidence found ({n_found} paper{'s' if n_found != 1 else ''})"
                if found else
                "✅ No contradictory evidence found in PubMed (25 years)"
            )
            color = "#E9A23B" if found else "#2e9e5b"
            st.markdown(
                f'<p style="font-size:0.82rem; color:{color}; '
                f'font-weight:600; margin:0.2rem 0">🔍 {label}</p>',
                unsafe_allow_html=True,
            )
            if found and papers:
                with st.expander("📄 Contradictory references", expanded=False):
                    for p in papers:
                        st.caption(_format_apa(p))

        # ── Inline numeric summary lines ──────────────────────────────────
        cited_pids     = hyp.get("sources_used", []) or []
        retrieved_pids = hyp.get("sources_retrieved", []) or []
        past_bullets   = [b.strip("• ").strip() for b in hyp.get("past_summary",   "").split("\n") if b.strip()]
        future_bullets = [b.strip("• ").strip() for b in hyp.get("future_summary", "").split("\n") if b.strip()]
        total_bullets  = len(past_bullets) + len(future_bullets)

        st.markdown(
            f'<p class="summary-line">{len(cited_pids)} sources cited · '
            f'{len(retrieved_pids)} retrieved</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="summary-line">gap score {hyp.get("gap_score", 0.0):.2f} '
            f'· {total_bullets} bullets</p>',
            unsafe_allow_html=True,
        )

        # ── Sources expander (closed by default) ─────────────────────────
        with st.expander("📚 Sources", expanded=False):
            # Build a quick PMID → paper dict from cat_papers for lookup.
            pid_to_paper: dict[str, dict] = {}
            for cat, papers in (cat_papers or {}).items():
                for p in papers:
                    pid = p.get("pmid") or p.get("paper_id", "")
                    if pid:
                        pid_to_paper[pid] = p
            # Path A: resolve uploaded-PDF ids to their readable titles (same
            # source the PDF export uses), instead of showing raw temp filenames.
            for p in st.session_state.get("_local_paper_metadata", []):
                pid = p.get("paper_id", "")
                if pid and pid not in pid_to_paper:
                    pid_to_paper[pid] = p

            # Always surface the user's uploaded paper(s) first, regardless of
            # whether they were cited or appear in the PubMed retrieved set — the
            # user uploaded them and expects to see them used.
            local_meta = st.session_state.get("_local_paper_metadata", [])
            if local_meta and st.session_state.get("path_choice") in ("combined", "local_only"):
                st.markdown(f"**📄 Your uploaded paper(s) ({len(local_meta)})**")
                for p in local_meta:
                    _render_apa_source(p, idx, p.get("paper_id", ""))
                st.divider()

            # Cited subset (first tier)
            if cited_pids:
                st.markdown(f"**Cited in this hypothesis ({len(cited_pids)})**")
                for pid in cited_pids:
                    paper = pid_to_paper.get(pid, {"title": pid, "year": "n.d.",
                                                    "authors": [], "journal": ""})
                    _render_apa_source(paper, idx, pid)
            else:
                st.caption("No specific citations recorded for this hypothesis.")

            # All retrieved (nested expander)
            # Rating + comment row:
            #   - Count reflects only the papers SHOWN in this expander
            #     (not the total across all 6 categories).
            #   - For H2..H6 (complement non-empty), show ONLY the complementary
            #     category's papers — primary papers are redundant since
            #     they're already represented in "Cited in this hypothesis"
            #     and shown in H1's expander.
            #   - For H1 (no complement), show primary (the only papers used).
            if cat_papers:
                cats_to_show = comp if comp else [primary]
                shown_count  = sum(
                    len(cat_papers.get(c, [])) for c in cats_to_show
                )
                if shown_count > 0:
                    st.divider()
                    header_text = (
                        f"Show all retrieved papers from complementary category "
                        f"({shown_count})"
                        if comp
                        else f"Show all retrieved papers from primary category "
                             f"({shown_count})"
                    )
                    with st.expander(header_text, expanded=False):
                        for cat in cats_to_show:
                            papers = cat_papers.get(cat, [])
                            if not papers:
                                continue
                            rel = (cat_relevance or {}).get(cat, {})
                            badge = ""
                            if rel.get("low_relevance_badge"):
                                badge = " ⚠ low-relevance retrieval"
                            st.markdown(
                                f"**{cat} ({len(papers)})**"
                                f"{badge}"
                            )
                            for p in papers:
                                pid = p.get("pmid") or p.get("paper_id", "")
                                _render_apa_source(p, idx, pid)

        # ── Gap analysis expander (closed by default) ─────────────────────
        with st.expander(
            f"📖 Gap analysis — score {hyp.get('gap_score', 0.0):.2f}",
            expanded=False,
        ):
            if past_bullets or future_bullets:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Past — what has been done**")
                    for b in past_bullets:
                        st.markdown(f"• {b}")
                with c2:
                    st.markdown("**Future — what is recommended**")
                    for b in future_bullets:
                        st.markdown(f"• {b}")
                st.caption(
                    f"Score = 1 − cosine(past_summary, future_summary) = "
                    f"{hyp.get('gap_score', 0.0):.2f}. "
                    f"Higher = bigger divergence = more room for new hypotheses."
                )
            else:
                st.caption("Gap analysis content not available for this hypothesis.")

        # ── Suggested approach (small, always visible at bottom) ──────────
        approach = hyp.get("suggested_approach") or []
        if approach:
            st.markdown("**Suggested approach to test:**")
            if isinstance(approach, list):
                for a in approach:
                    st.markdown(f"• {a}")
            else:
                st.markdown(f"• {approach}")

        # ── Feedback form ─────
        if show_feedback_form:
            st.markdown(
                "<hr style='margin: 0.4rem 0'>",
                unsafe_allow_html=True,
            )
            # One row: rating | comment | Continue | Stop
            # Column proportions: rate (4) · comment (4) · Continue (2) · Stop (2)
            # Rate gets extra width so all 5 radio options stay on one line.
            rc1, rc2, rc3, rc4 = st.columns([4, 4, 2, 2], gap="small")
            with rc1:
                rating = st.radio(
                    "Rate (1–5)",
                    options=[1, 2, 3, 4, 5],
                    index=None,         # No preselection — user must choose
                    horizontal=True,
                    key=f"rating_{idx}",
                )
            with rc2:
                comment = st.text_input(
                    "Comment (optional)",
                    placeholder="What do you think?",
                    key=f"comment_{idx}",
                )

            # Continue button stays disabled (grey) until the user picks a rating.
            enter_disabled = (rating is None)
            is_last        = idx >= config.MAX_HYPOTHESES - 1

            with rc3:
                st.markdown(
                    "<div style='height: 1.6rem'></div>",
                    unsafe_allow_html=True,
                )
                if not is_last:
                    if st.button(
                        "Continue",
                        type="primary",
                        key=f"continue_{idx}",
                        use_container_width=True,
                        disabled=enter_disabled,
                        help=(
                            "Select a rating (1–5) first"
                            if enter_disabled else
                            "Continue to next hypothesis"
                        ),
                    ):
                        feedback = {"rating": rating, "comment": comment, "decision": "continue"}
                else:
                    # Last hypothesis — "Finish" replaces Enter; requires rating
                    if st.button(
                        "Finish & Get Report ↓",
                        type="primary",
                        key=f"finish_{idx}",
                        use_container_width=True,
                        disabled=enter_disabled,
                        help=(
                            "Select a rating (1–5) first"
                            if enter_disabled else
                            "Submit rating and go to download"
                        ),
                    ):
                        feedback = {"rating": rating, "comment": comment, "decision": "stop"}

            with rc4:
                st.markdown(
                    "<div style='height: 1.6rem'></div>",
                    unsafe_allow_html=True,
                )
                # Stop only shown on non-last hypotheses so user can exit early
                if not is_last:
                    if st.button(
                        "Stop",
                        key=f"stop_{idx}",
                        use_container_width=True,
                    ):
                        feedback = {
                            "rating":   rating if rating is not None else 0,
                            "comment":  comment,
                            "decision": "stop",
                        }

            # Hint text
            if enter_disabled:
                if is_last:
                    st.caption(
                        "👆 Choose a rating to enable **Finish & Get Report ↓**."
                    )
                else:
                    st.caption(
                        "👆 Choose a rating to enable the **Continue** button "
                        "and continue to the next hypothesis."
                    )

        st.markdown("</div>", unsafe_allow_html=True)

    return feedback


def _render_apa_source(paper: dict, idx: int, pid: str) -> None:
    """
    Render one APA citation line.

    If metadata is present, the APA line itself
    is the expander label (click to expand for metadata).  No separate ⓘ
    button.  If no metadata, render as a plain bullet.
    """
    apa  = _format_apa(paper)
    meta = paper.get("metadata")

    if meta:
        # The APA citation itself is the expander label.  Click it
        # (the chevron) to see structured metadata inline below.
        with st.expander(apa, expanded=False):
            st.markdown(f"**Topic:** {meta.get('topic', 'Not available')}")
            st.markdown(f"**Methods:** {meta.get('methods', 'Not available')}")
            st.markdown(
                f"**Key findings:** {meta.get('key_findings', 'Not available')}"
            )
            st.markdown(
                f"**Limitations:** {meta.get('limitations', 'Not available')}"
            )
            st.markdown(
                f"**Future:** "
                f"{meta.get('future_recommendations', 'Not available')}"
            )
    else:
        # No metadata available — plain bullet, no clickable element.
        st.markdown(
            f'<p class="apa-source">• {apa}</p>',
            unsafe_allow_html=True,
        )


# =============================================================================
# Sidebar — cost meter + developer panel  (v2.1: simplified)
# =============================================================================

def _render_sidebar() -> None:
    with st.sidebar:

        # ── About (always at top, expanded on first visit) ─────────────────
        with st.expander("ℹ️ About Neurohypothesis", expanded=False):
            st.markdown(
                "**Neurohypothesis** generates evidence-grounded research "
                "hypotheses from your uploaded PDFs, live PubMed search, or both."
            )
            st.markdown("**Three ways to generate:**")
            st.markdown(
                "- **PDFs only (Path A)** — builds past→future research gaps "
                "from the papers you upload and proposes hypotheses grounded in "
                "them, **one at a time (up to 6)**: rate each, then Continue to "
                "generate the next or Finish to export. Each hypothesis is anchored "
                "on a different past↔future gap pair for variety. Intentionally "
                "*not* cross-scale — it stays within your uploaded literature and "
                "isn't organised by the 6 method categories.\n"
                "- **PubMed only (Path B)** — searches live PubMed across the 6 "
                "method categories and produces up to 6 cross-scale hypotheses "
                "(H1 anchored to the primary category, H2–H6 pairing it with "
                "complementary ones).\n"
                "- **Combined (Path C)** — uses PubMed's category structure as the "
                "backbone and supplements the evidence with your uploaded PDFs."
            )
            st.caption(
                "Every path scores each hypothesis the same way: originality vs the "
                "evidence it was built from, plausibility (6-dimension judge), and a "
                "PubMed novelty check against up to 25 years of prior literature "
                "(nearest-neighbour cosine — lower similarity means more novel)."
            )
            st.markdown("**How to use:**")
            st.markdown(
                "1. Enter a specific research question\n"
                "2. Optionally upload up to 3 PDFs\n"
                "3. Choose a path: PubMed only, PDFs only, or Combined\n"
                "4. Rate each hypothesis (1–5 ⭐) before continuing\n"
                "5. Download the PDF report at the end"
            )
            st.markdown("**PubMed search:**")
            st.caption(
                "Each run searches PubMed across all 6 neuroscience categories, "
                "retrieving up to **10 abstracts per category** (60 total) "
                "from the **last 25 years**. Abstracts are ranked by semantic "
                "similarity to your query and filtered for relevance before "
                "hypothesis generation."
            )
            st.markdown("**The 6 search categories:**")
            for cat in config.CATEGORIES:
                st.caption(f"• {cat}")
            st.markdown("**Score meanings:**")
            st.caption(
                "**Originality** — semantic distance from past literature.  "
                "**Plausibility** — 6-dimension LLM judge (novelty, testability, "
                "mechanistic coherence, citation traceability, conflict awareness, "
                "usefulness).  "
                "**Gap** — distance between past findings and future directions.  "
                "**PubMed-check** — freshness of supporting literature (last 5 years)."
            )
            st.caption("⏱ Typical processing time: 2–4 minutes per run.")
            st.caption("⚠️ AI outputs require expert validation before use in research.")

        st.divider()

        # ── Lock sidebar during generation to prevent rerun interference ───
        if st.session_state.get("phase") == "running":
            st.info("⏳ Generating… please wait until the first hypothesis appears.")
            return

        # ── Session stats ──────────────────────────────────────────────────
        st.header("📊 Session stats")
        tracker = get_tracker()
        c1, c2 = st.columns(2)
        c1.metric("Input tokens",  f"{tracker.total_input_tokens:,}")
        c2.metric("Output tokens", f"{tracker.total_output_tokens:,}")
        st.metric("Estimated cost (USD)", f"${tracker.total_cost_usd:.5f}")

        st.divider()

        # ── Graph topology (visible to all users) ──────────────────────────
        graph_png = config.EXPORTS_DIR / "graph_topology.png"
        graph_md  = config.EXPORTS_DIR / "graph_topology.md"
        if graph_png.exists() or graph_md.exists():
            with st.expander("📊 Graph topology"):
                if graph_png.exists():
                    st.image(
                        str(graph_png),
                        caption="LangGraph agent topology",
                        use_container_width=True,
                    )

        st.divider()

        # ── Developer ──────────────────────────────────────────────────────
        with st.expander("🔧 Developer", expanded=False):

            n_calls = len(tracker.calls)
            with st.expander(f"🔬 Per-node cost ({n_calls} LLM calls)", expanded=False):
                if n_calls == 0:
                    st.caption("No LLM calls tracked yet — run a session first.")
                else:
                    breakdown = tracker.node_breakdown()
                    for node, row in breakdown.items():
                        label = _NODE_LABELS.get(node, node)
                        st.caption(
                            f"**{label}**  \n"
                            f"in={row['input_tokens']:,} "
                            f"out={row['output_tokens']:,} "
                            f"· ${row['cost_usd']:.6f}"
                        )

            errors = st.session_state.get("_run_errors", [])
            if errors:
                with st.expander(f"⚠️ Errors ({len(errors)})"):
                    for err in errors:
                        icon = "✅" if err.get("recovered") else "❌"
                        st.caption(
                            f"{icon} **[{err.get('node', '?')}]** "
                            f"{err.get('type', '')} — "
                            f"{err.get('message', '')[:120]}"
                        )

            timings = st.session_state.get("_node_timings", {})
            with st.expander(f"⏱ Node timings ({len(timings)} nodes)", expanded=False):
                if not timings:
                    st.caption("No timings yet — run a session first.")
                else:
                    for node, secs in sorted(timings.items(), key=lambda x: -x[1]):
                        label = _NODE_LABELS.get(node, node)
                        st.caption(f"{label}: **{secs:.2f}s**")

            st.divider()
            st.caption(
                f"Session: `{st.session_state.session_id[:8]}…`  \n"
                f"User: `{st.session_state.user_id[:8]}…`"
            )


# =============================================================================
# Graph streaming  (v2.1: inline reformulation lines per category)
# =============================================================================

def _sync_debug_state(graph_state) -> None:
    """
    Sync node_timings and errors from the final graph state into session state.
    This is a fallback for cases where the streaming loop missed updates
    (e.g. parallel worker threads cannot write to st.session_state directly).
    """
    graph_timings = graph_state.values.get("node_timings", {})
    if graph_timings:
        existing = st.session_state.get("_node_timings", {})
        existing.update(graph_timings)
        st.session_state["_node_timings"] = existing

    graph_errors = graph_state.values.get("errors", [])
    if graph_errors:
        existing  = st.session_state.get("_run_errors", [])
        seen_msgs = {e.get("message", "") for e in existing}
        for err in graph_errors:
            if err.get("message", "") not in seen_msgs:
                existing.append(err)
        st.session_state["_run_errors"] = existing


_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth", "ninth", "tenth"]

def _ordinal(n: int) -> str:
    """Return spelled-out ordinal for n (1-indexed). Falls back to '#{n}'."""
    return _ORDINALS[n - 1] if 1 <= n <= len(_ORDINALS) else f"#{n}"


def _stream_graph(input_or_command, thread: dict, progress_placeholder, label: str = "Running analysis…") -> tuple[dict, list[str]]:
    """
    Stream the graph; surface per-category reformulations inline as they happen.
    """
    graph = _get_graph()
    lines: list[str] = []
    latest: dict = {}

    with progress_placeholder.status(label, expanded=True) as status:
        try:
            for chunk in graph.stream(
                input_or_command,
                thread,
                stream_mode="updates",
            ):
                for node_name, partial in chunk.items():
                    if node_name.startswith("__"):
                        continue
                    label = _NODE_LABELS.get(node_name, node_name)

                    # Special handling for per-category retrieval — surface
                    # each category's reformulation inline so the user sees
                    # what was asked of PubMed.
                    if (
                        node_name == "n5_per_category_retrieve"
                        and partial
                        and "category_reformulations" in partial
                    ):
                        st.write(f"✓ {label}")
                        for cat, attempts in (partial["category_reformulations"] or {}).items():
                            if attempts:
                                last = attempts[-1]
                                refo = last.get("reformulation", "")
                                npass = last.get("n_relevant", 0)
                                # Issue 2 (May 11): three-state outcome instead
                                # of the misleading "keep n_relevant=0".
                                if last.get("passed"):
                                    outcome = f"PASS, {npass} papers"
                                elif npass > 0:
                                    outcome = f"low relevance, kept {npass} papers"
                                else:
                                    outcome = "no papers found after 3 attempts"
                                st.markdown(
                                    f'<p class="reformulation">'
                                    f'→ <b>{cat}</b>: "{refo}" '
                                    f'<span style="color:#999">({outcome})</span>'
                                    f'</p>',
                                    unsafe_allow_html=True,
                                )
                                lines.append(f"  → {cat}: {refo} — {outcome}")

                    elif node_name == "n5_pick_primary" and partial and "primary_category" in partial:
                        st.write(f"✓ {label}: **{partial['primary_category']}**")
                        lines.append(f"primary={partial['primary_category']}")

                    elif node_name == "n5_order_categories":
                        # Issue 4(a) (May 11): the next node (per-category
                        # retrieval) runs silently for 2-3 minutes.  Emit an
                        # explicit placeholder so the user knows what's
                        # happening, instead of staring at "Ordering categories"
                        # for 3 minutes.
                        st.write(f"✓ {label}")
                        ordered = partial.get("ordered_categories", []) if partial else []
                        _pdf_note = (
                            f'<p style="color:#8a6cb4; font-style:italic; '
                            f'margin-left:1.2rem;">'
                            f'📄 Your uploaded paper(s) are ingested and merged into '
                            f'the evidence for every hypothesis.</p>'
                            if st.session_state.get("path_choice") == "combined"
                            else ""
                        )
                        st.markdown(
                            f'<p style="color:#8a6cb4; font-style:italic; '
                            f'margin-left:1.2rem;">'
                            f'⏳ Now searching PubMed across '
                            f'{len(ordered) or 6} categories. This typically '
                            f'takes <b>2–4 minutes</b>; please be patient!'
                            f'</p>'
                            f'{_pdf_note}',
                            unsafe_allow_html=True,
                        )
                        lines.append("[ordering done — awaiting PubMed search]")

                    else:
                        st.write(f"✓ {label}")
                        lines.append(label)

                    latest.update(partial or {})

                    timings = st.session_state.get("_node_timings", {})
                    timings.update(partial.get("node_timings", {}))
                    st.session_state["_node_timings"] = timings

                    errs = st.session_state.get("_run_errors", [])
                    errs.extend(partial.get("errors", []))
                    st.session_state["_run_errors"] = errs

            status.update(label="Analysis complete ✓", state="complete")
        except Exception as exc:
            status.update(label=f"Error: {exc}", state="error")
            logger.exception(f"Graph stream error: {exc}")

    return latest, lines


# =============================================================================
# Main app
# =============================================================================

def _render_disclaimer() -> None:
    """In-flow disclaimer block — sits at the bottom of the main content,
    aligns with the title, and flexes with the sidebar."""
    st.markdown(
        f'<div class="disclaimer-footer">'
        f'<span class="warn">⚠ Disclaimer</span> &nbsp; {config.DISCLAIMER}'
        f'</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    _init_logging()
    _init_db()
    _init_session()
    graph = _get_graph()

    _render_sidebar()

    st.title("Neurohypothesis")
    st.caption(config.APP_SUBTITLE)

    st.divider()

    # =========================================================================
    # Phase: INPUT
    # =========================================================================
    if st.session_state.phase == "input":
        # 3-column layout, Generate in the MIDDLE
        #   Col 1 (wide):   research question (left)
        #   Col 2 (mid):    path choice + conditional PDF upload (middle)
        #   Col 3 (narrow): Generate button (right), disabled until ready
        col1, col2, col3 = st.columns([5, 3, 2], gap="small")

        uploaded_files: list = []

        with col1:
            st.markdown("**1 · Research question**")
            topic = st.text_area(
                "Research question",
                placeholder="e.g. How cardiovascular risk factors affect brain white matter?",
                height=120,
                label_visibility="collapsed",
                key="input_topic",
            )
            char_count = len(topic.strip())
            st.caption(
                f"{char_count}/{config.MAX_QUERY_LENGTH} characters "
                f"(min {config.MIN_QUERY_LENGTH})"
            )

        with col2:
            st.markdown("**2 · Choose how to generate**")
            radio_choice = st.radio(
                "Path",
                options=["PubMed only", "PDFs only", "PDFs + PubMed"],
                index=0,
                label_visibility="collapsed",
                key="input_path",
            )
            if radio_choice == "PDFs only":
                path_choice = "local_only"
            elif radio_choice == "PDFs + PubMed":
                path_choice = "combined"
            else:
                path_choice = "pubmed_only"

            needs_pdf = path_choice in ("local_only", "combined")
            if needs_pdf:
                uploaded_files = st.file_uploader(
                    f"Upload up to {config.MAX_PDF_COUNT} PDFs",
                    type=["pdf"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                    key="input_pdfs",
                ) or []
                if len(uploaded_files) > config.MAX_PDF_COUNT:
                    st.warning(
                        f"Only the first {config.MAX_PDF_COUNT} will be used. "
                        f"You uploaded {len(uploaded_files)}."
                    )
                    uploaded_files = uploaded_files[:config.MAX_PDF_COUNT]
                st.caption(
                    f"✅ {len(uploaded_files)} PDF(s) ready" if uploaded_files
                    else f"Upload at least 1 PDF for “{radio_choice}”."
                )
            else:
                st.caption("No upload needed — searching PubMed.")

        with col3:
            st.markdown("**3 · Generate**")
            ready = (
                char_count >= config.MIN_QUERY_LENGTH
                and (not needs_pdf or len(uploaded_files) > 0)
            )
            submitted = st.button(
                "Generate Hypothesis",
                type="primary",
                key="btn_generate",
                use_container_width=False,
                disabled=not ready,
            )
            if not ready:
                if char_count < config.MIN_QUERY_LENGTH:
                    st.caption("✍️ Enter your question first.")
                elif needs_pdf and not uploaded_files:
                    st.caption("📄 Upload at least one PDF.")

        if submitted:
            if char_count < config.MIN_QUERY_LENGTH:
                st.error(
                    f"Please enter a topic of at least "
                    f"{config.MIN_QUERY_LENGTH} characters."
                )
            else:
                tmp_paths: list[str] = []
                for uf in (uploaded_files or []):
                    suffix   = Path(uf.name).suffix
                    safe_stem = Path(uf.name).stem[:40].replace(" ", "_")
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix, prefix=f"{safe_stem}_"
                    ) as tmp:
                        tmp.write(uf.read())
                        tmp_paths.append(tmp.name)

                st.session_state.pdf_tmp_paths = tmp_paths
                st.session_state.topic         = topic.strip()
                st.session_state.path_choice   = path_choice
                st.session_state.phase         = "running"

                # Explicitly clear input-phase widget state
                # so Streamlit unmounts the file_uploader's native <input
                # type="file"> element cleanly.  Mitigates the reported
                # "clicking during running phase opens file picker" bug.
                # Also clears the _running_started flag (in case a prior
                # run was aborted mid-flight).
                for stale_key in (
                    "input_topic", "input_pdfs", "input_path",
                    "_running_started",
                ):
                    if stale_key in st.session_state:
                        try:
                            del st.session_state[stale_key]
                        except Exception:
                            pass

                from src.db import create_session, upsert_user
                upsert_user(st.session_state.user_id)
                create_session(
                    session_id=st.session_state.session_id,
                    user_id=st.session_state.user_id,
                    topic=topic.strip(),
                )
                get_tracker().set_session(
                    st.session_state.user_id,
                    st.session_state.session_id,
                )
                st.rerun()

    # =========================================================================
    # Phase: RUNNING (graph streams to first interrupt)
    # =========================================================================
    elif st.session_state.phase == "running":
        st.subheader(f"Topic: *{st.session_state.topic}*")
        st.caption(
            f"Path: **{st.session_state.path_choice}** — the agent is preparing "
            "your first hypothesis…"
        )

        thread = {"configurable": {"thread_id": st.session_state.session_id}}
        st.session_state.thread = thread

        # Guard against widget-click reruns inside the
        # running phase.  Streamlit re-executes the script from the top on
        # ANY widget interaction, which means clicking a sidebar button
        # while the graph is generating would invoke _stream_graph a second
        # time — hitting N1's throttle limit ("Please wait 5 seconds before
        # starting a new run").  The _running_started flag ensures the
        # initial graph invocation happens exactly once per session.
        if not st.session_state.get("_running_started"):
            st.session_state["_running_started"] = True

            initial_state = {
                "user_id":                   st.session_state.user_id,
                "session_id":                st.session_state.session_id,
                "topic":                     st.session_state.topic,
                "pdf_paths":                 st.session_state.pdf_tmp_paths,
                "has_uploaded_pdfs":         bool(st.session_state.pdf_tmp_paths),
                "path_choice":               st.session_state.path_choice,
                "current_hypothesis_index":  0,
                "hypotheses":                [],
                "quality_gate_attempts":     {},
                "errors":                    [],
                "node_timings":              {},
                "token_usage":               {},
                "local_paper_metadata":      [],
                "category_papers":           {},
                "category_relevance":        {},
                "category_reformulations":   {},
            }

            progress_box = st.empty()
            _gen_label = (
                "🔍 Building the gap from your PDFs & generating hypotheses…"
                if st.session_state.path_choice == "local_only"
                else "🔍 Searching PubMed & generating first hypothesis…"
            )
            _render_disclaimer()   # visible during the long blocking stream
            _stream_graph(initial_state, thread, progress_box, label=_gen_label)
        else:
            st.info(
                "⏳ Generating hypotheses — this takes 2–4 minutes. "
                "Feel free to browse the sidebar while waiting."
            )
            _render_disclaimer()

        graph_state = graph.get_state(thread)
        _sync_debug_state(graph_state)
        hyps = graph_state.values.get("hypotheses", [])

        if not graph_state.values.get("validation_passed", True):
            errors = graph_state.values.get("errors", [])
            msg = errors[-1].get("message", "Validation failed.") if errors else "Validation failed."
            st.warning(msg, icon="⚠️")
            if st.button("Try again"):
                st.session_state.phase = "input"
                st.rerun()

        elif graph_state.next and hyps:
            st.session_state.hypotheses = hyps
            st.session_state["_category_papers"]       = graph_state.values.get("category_papers", {})
            st.session_state["_category_relevance"]    = graph_state.values.get("category_relevance", {})
            st.session_state["_local_paper_metadata"]  = graph_state.values.get("local_paper_metadata", [])
            st.session_state.phase = "feedback"
            st.rerun()

        elif graph_state.next and not hyps:
            # Graph still running — poll quickly without blocking the UI long
            import time
            time.sleep(0.5)
            st.rerun()

        else:
            st.session_state.hypotheses = hyps
            st.session_state["_category_papers"]       = graph_state.values.get("category_papers", {})
            st.session_state["_category_relevance"]    = graph_state.values.get("category_relevance", {})
            st.session_state["_local_paper_metadata"]  = graph_state.values.get("local_paper_metadata", [])
            st.session_state.phase = "done"
            st.rerun()

    # =========================================================================
    # Phase: FEEDBACK (hypothesis card + rating)
    # =========================================================================
    elif st.session_state.phase == "feedback":
        thread = st.session_state.thread
        if thread is None:
            st.error("Session state lost — please restart.")
            if st.button("Start over"):
                st.session_state.phase = "input"
                st.rerun()
            st.stop()

        hypotheses     = st.session_state.hypotheses
        cat_papers     = st.session_state.get("_category_papers",    {})
        cat_relevance  = st.session_state.get("_category_relevance", {})
        prev_hyps      = hypotheses[:-1] if len(hypotheses) > 1 else []
        curr_hyp       = hypotheses[-1] if hypotheses else {}

        if prev_hyps:
            st.subheader("Previous hypotheses")
            for h in prev_hyps:
                _render_hypothesis_card(
                    h,
                    cat_relevance=cat_relevance,
                    cat_papers=cat_papers,
                    show_feedback_form=False,
                )
            st.divider()

        # The card itself shows "Hypothesis N · Category" in its blue
        # header bar, so no separate subheader above is needed.
        feedback = _render_hypothesis_card(
            curr_hyp,
            cat_relevance=cat_relevance,
            cat_papers=cat_papers,
            show_feedback_form=True,
        )

        if feedback:
            progress_box = st.empty()
            next_n = len(st.session_state.hypotheses) + 1
            _stream_graph(Command(resume=feedback), thread, progress_box,
                          label=f"Generating {_ordinal(next_n)} hypothesis…")

            graph_state = graph.get_state(thread)
            _sync_debug_state(graph_state)
            new_hyps = graph_state.values.get("hypotheses", [])
            if len(new_hyps) > len(st.session_state.hypotheses):
                st.session_state.hypotheses = new_hyps

            if graph_state.next:
                st.session_state.phase = "feedback"
            else:
                st.session_state.phase = "done"
            st.rerun()

    # =========================================================================
    # Phase: DONE
    # =========================================================================
    elif st.session_state.phase == "done":
        n_hyps = len(st.session_state.hypotheses)
        st.success(f"✅ Session complete — {n_hyps} hypothesis{'es' if n_hyps != 1 else ''} generated.")
        st.subheader(f"Topic: *{st.session_state.get('topic', '')}*")
        st.divider()

        # ── PDF download — PRIMARY ACTION, always visible at top ───────────
        st.subheader("📥 Download Report")
        try:
            pdf_bytes = render_pdf_bytes(
                topic=st.session_state.get("topic", ""),
                hypotheses=st.session_state.hypotheses,
                cat_papers=st.session_state.get("_category_papers", {}),
                local_paper_metadata=st.session_state.get("_local_paper_metadata", []),
            )
            topic_slug = (
                st.session_state.get("topic", "hypothesis")[:40]
                .replace(" ", "_")
                .replace("/", "-")
            )
            st.download_button(
                label="⬇️ Download PDF report",
                data=pdf_bytes,
                file_name=f"Neurohypothesis_{topic_slug}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=False,
            )
        except Exception as exc:
            st.warning(f"PDF generation failed: {exc}")

        st.divider()

        # ── Hypothesis review — collapsed by default ───────────────────────
        with st.expander(f"📋 Review all {n_hyps} hypotheses", expanded=False):
            cat_papers    = st.session_state.get("_category_papers",    {})
            cat_relevance = st.session_state.get("_category_relevance", {})
            for hyp in st.session_state.hypotheses:
                _render_hypothesis_card(
                    hyp,
                    cat_relevance=cat_relevance,
                    cat_papers=cat_papers,
                    show_feedback_form=False,
                )
                rating  = hyp.get("user_rating")
                comment = hyp.get("user_comment", "")
                if rating:
                    stars = "★" * rating + "☆" * (5 - rating)
                    st.caption(f"Your rating: {stars} ({rating}/5)  {comment or ''}")
                st.divider()

        st.divider()
        if st.button("🔄 Start a new session", use_container_width=False):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        # ── Supabase session sync (non-blocking, silent) ───────────────────
        if not st.session_state.get("_feedback_logged"):
            tracker = get_tracker()
            synced = log_session(
                session_id  = st.session_state.session_id,
                user_id     = st.session_state.user_id,
                topic       = st.session_state.get("topic", ""),
                hypotheses  = st.session_state.hypotheses,
                cost_usd    = tracker.total_cost_usd,
                path_choice = st.session_state.get("path_choice", "pubmed_only"),
            )
            st.session_state["_feedback_logged"] = True
            if not synced:
                logger.warning("Supabase sync returned False — check credentials in .env")

    # ── Disclaimer (in-flow, bottom of content) ───────────────────────────
    # Aligns with the title and flexes with the sidebar because it shares the
    # main content box.  (The running phase renders its own copy, since it
    # blocks before reaching here.)
    _render_disclaimer()


# =============================================================================
# Entry
# =============================================================================

main()
