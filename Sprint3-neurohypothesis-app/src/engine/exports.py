"""
PDF export for Neurohypothesis v2 (N19 export_results).

Generates a clean A4 report containing all hypotheses produced in a
session, with per-hypothesis gap analysis, scores, evidence, and user
feedback.

Extended from v1 exports.py to handle up to 3 hypotheses per session.
The reportlab palette and style definitions are carried verbatim from v1.

Public API:
    - export_session_to_pdf(topic, hypotheses, session_id) -> Path
      Writes the PDF to config.EXPORTS_DIR and returns the file path.
    - render_pdf_bytes(topic, hypotheses) -> bytes
      Returns raw PDF bytes for st.download_button() in app.py.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from loguru import logger
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config

# =============================================================================
# Colour palette  (blue tones — matching UI palette from config)
# =============================================================================

BLUE_DARK   = colors.HexColor("#1a4f8a")
BLUE_MED    = colors.HexColor("#2e7bc4")
BLUE_LIGHT  = colors.HexColor("#7ab3e0")
GREY_LIGHT  = colors.HexColor("#f5f5f5")
GREY_BORDER = colors.HexColor("#dddddd")
AMBER       = colors.HexColor("#E9A23B")   # low-confidence badge
GREEN_SOFT  = colors.HexColor("#2e7d46")   # pass grade
BLACK       = colors.black
WHITE       = colors.white


# =============================================================================
# APA citation helper
# =============================================================================

def _format_apa(paper: dict) -> str:
    """
    Format a paper as a simple APA-style line, matching the in-app card style.

    Author, A., Author, B., & Author, C. (Year). Title. Journal.

    Falls back gracefully when fields are missing.  No volume/issue/pages/DOI
    (per v2.1 spec).
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
    tail = f" <i>{journal}</i>." if journal else ""
    return f"{lead}{title}.{tail}"


# =============================================================================
# Style definitions  (ported from v1, + hypothesis_num for H1/H2/H3 headers)
# =============================================================================

def _styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle(
            "title", fontSize=18, fontName="Helvetica-Bold",
            textColor=BLUE_DARK, spaceAfter=4, alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontSize=10, fontName="Helvetica",
            textColor=colors.HexColor("#555555"), spaceAfter=12,
        ),
        "section": ParagraphStyle(
            "section", fontSize=13, fontName="Helvetica-Bold",
            textColor=BLUE_DARK, spaceBefore=14, spaceAfter=6,
        ),
        "hyp_header": ParagraphStyle(
            "hyp_header", fontSize=14, fontName="Helvetica-Bold",
            textColor=WHITE, spaceAfter=0, spaceBefore=0,
        ),
        "body": ParagraphStyle(
            "body", fontSize=10, fontName="Helvetica",
            textColor=BLACK, spaceAfter=4, leading=14,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontSize=10, fontName="Helvetica",
            textColor=BLACK, spaceAfter=3, leftIndent=12, leading=14,
        ),
        "hypothesis": ParagraphStyle(
            "hypothesis", fontSize=11, fontName="Helvetica-Bold",
            textColor=BLACK, spaceAfter=6, leading=16,
        ),
        "grade": ParagraphStyle(
            "grade", fontSize=10, fontName="Helvetica",
            textColor=BLUE_MED, spaceAfter=4,
        ),
        "badge": ParagraphStyle(
            "badge", fontSize=9, fontName="Helvetica-Bold",
            textColor=AMBER, spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption", fontSize=9, fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#777777"), spaceAfter=4,
        ),
        "feedback": ParagraphStyle(
            "feedback", fontSize=10, fontName="Helvetica",
            textColor=GREEN_SOFT, spaceAfter=4,
        ),
    }


def _hr(thick: float = 0.5) -> HRFlowable:
    return HRFlowable(
        width="100%", thickness=thick,
        color=GREY_BORDER, spaceAfter=6, spaceBefore=6,
    )


def _coloured_table_row(text: str, bg: Any, style: ParagraphStyle) -> Table:
    """Render a coloured header row for hypothesis blocks."""
    tbl = Table([[Paragraph(text, style)]], colWidths=["100%"])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    return tbl


# =============================================================================
# Per-hypothesis section builder
# =============================================================================

def _build_hypothesis_section(
    hyp:           dict[str, Any],
    idx:           int,
    S:             dict[str, ParagraphStyle],
    doc_width:     float,
    pid_to_paper:  dict[str, dict] | None = None,
) -> list:
    """
    Build the reportlab story elements for one hypothesis.

    When `pid_to_paper` is provided, "Supporting evidence" is
    rendered as APA citations instead of raw PMIDs.

    Args:
        hyp:          Hypothesis dict from AgentState (matches Hypothesis TypedDict).
        idx:          0-based index (displayed as H1/H2/H3).
        S:            Style dict from _styles().
        doc_width:    usable page width for table column calculations.
        pid_to_paper: optional lookup {pmid -> paper_dict_with_metadata}
                      used to render APA citations.

    Returns:
        List of reportlab flowables.
    """
    pid_to_paper = pid_to_paper or {}
    story = []
    label = f"Hypothesis {idx + 1}"
    primary  = hyp.get("primary_category", "")
    comp     = hyp.get("complementary_categories", [])
    cat_text = f"{primary}" + (f" × {', '.join(comp)}" if comp else "")
    header_text = f"{label}  ·  {cat_text}" if cat_text.strip() else label

    # ── Coloured hypothesis header bar ────────────────────────────────────
    story.append(Spacer(1, 0.4 * cm))
    story.append(_coloured_table_row(
        header_text,
        BLUE_DARK,
        S["hyp_header"],
    ))
    story.append(Spacer(1, 0.2 * cm))

    # ── Low-confidence badge ───────────────────────────────────────────────
    if hyp.get("low_confidence"):
        story.append(Paragraph(
            "⚠ Best-of-3 — quality gate not fully met after 3 attempts.",
            S["badge"],
        ))

    # ── Hypothesis statement ──────────────────────────────────────────────
    story.append(Paragraph(hyp.get("text", ""), S["hypothesis"]))

    # ── Scores ────────────────────────────────────────────────────────────
    orig_score = hyp.get("originality_score")
    orig_grade = hyp.get("originality_grade", "")
    plaus_avg  = hyp.get("plausibility_avg")
    gap_score  = hyp.get("gap_score")

    if orig_score is not None:
        story.append(Paragraph(
            f"Originality: {orig_score:.2f} ({orig_grade.replace('_', ' ')})",
            S["grade"],
        ))
    if plaus_avg is not None:
        story.append(Paragraph(
            f"Plausibility: {plaus_avg:.2f}/5.0",
            S["grade"],
        ))
    if gap_score is not None:
        story.append(Paragraph(
            f"Literature gap score: {gap_score:.3f}",
            S["caption"],
        ))

    # ── How to strengthen this hypothesis (mirrors the front-end UI) ──────
    # The UI shows improvement tips for weak dimensions, NOT the raw verdict;
    # the exported PDF must match what the user saw on screen.
    tips = hyp.get("improvement_tips") or []
    if tips:
        story.append(Spacer(1, 0.1 * cm))
        story.append(Paragraph("How to strengthen this hypothesis:", S["body"]))
        for tip in tips:
            story.append(Paragraph(f"• {tip}", S["bullet"]))

    story.append(Spacer(1, 0.2 * cm))

    # ── Past / future summaries table ─────────────────────────────────────
    past_raw   = hyp.get("past_summary", "")
    future_raw = hyp.get("future_summary", "")
    past_bullets   = [b.strip("• ").strip() for b in past_raw.split("\n")   if b.strip()]
    future_bullets = [b.strip("• ").strip() for b in future_raw.split("\n") if b.strip()]

    if past_bullets or future_bullets:
        max_rows  = max(len(past_bullets), len(future_bullets), 1)
        past_pad   = past_bullets   + [""] * (max_rows - len(past_bullets))
        future_pad = future_bullets + [""] * (max_rows - len(future_bullets))
        col_w      = (doc_width - 0.5 * cm) / 2
        tbl_data   = [[
            Paragraph("<b>Past — What Has Been Done</b>", S["body"]),
            Paragraph("<b>Future — What Is Recommended</b>", S["body"]),
        ]]
        for p, f in zip(past_pad, future_pad):
            tbl_data.append([Paragraph(p, S["bullet"]), Paragraph(f, S["bullet"])])

        tbl = Table(tbl_data, colWidths=[col_w, col_w])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GREY_LIGHT),
            ("GRID",       (0, 0), (-1, -1), 0.4, GREY_BORDER),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.2 * cm))

    # ── Supporting papers ─────
    sources = hyp.get("sources_used", [])
    if sources:
        story.append(Paragraph("Supporting evidence:", S["body"]))
        for src in sources[:10]:
            paper = pid_to_paper.get(str(src)) or pid_to_paper.get(src)
            if paper:
                apa = _format_apa(paper)
                story.append(Paragraph(f"• {apa}", S["bullet"]))
            else:
                # Fall back to raw identifier if we can't resolve to a paper
                # (e.g. a local PDF chunk paper_id that's not in cat_papers).
                story.append(Paragraph(f"• {src}", S["bullet"]))
        story.append(Spacer(1, 0.1 * cm))

    # ── Suggested approach ────────────────────────────────────────────────
    approach = hyp.get("suggested_approach", [])
    if isinstance(approach, str):
        approach = [approach] if approach.strip() else []
    if approach:
        story.append(Paragraph("Suggested approach to test:", S["body"]))
        for step in approach:
            story.append(Paragraph(f"• {step}", S["bullet"]))
        story.append(Spacer(1, 0.1 * cm))

    # ── User feedback ─────────────────────────────────────────────────────
    rating  = hyp.get("user_rating")
    comment = hyp.get("user_comment", "")
    if rating:
        stars = "★" * rating + "☆" * (5 - rating)
        story.append(Paragraph(f"Your rating: {stars}  ({rating}/5)", S["feedback"]))
    if comment and comment.strip():
        story.append(Paragraph(f"Your comment: {comment}", S["feedback"]))

    return story


# =============================================================================
# Main export functions
# =============================================================================

def render_pdf_bytes(
    topic:      str,
    hypotheses: list[dict[str, Any]],
    cat_papers: dict[str, list[dict]] | None = None,
    local_paper_metadata: list[dict] | None = None,
) -> bytes:
    """
    Render the session report as raw PDF bytes.

    Args:
        topic:       the user's original research topic.
        hypotheses:  list of Hypothesis dicts from AgentState["hypotheses"].
        cat_papers:  optional {category -> [paper_dict]} from category_papers.
        local_paper_metadata: optional list of local PDF paper dicts.

    Returns:
        PDF bytes — pass directly to st.download_button(data=...).
    """
    # Build a flat paper_id -> paper lookup from PubMed and local PDFs
    pid_to_paper: dict[str, dict] = {}
    if cat_papers:
        for papers in cat_papers.values():
            for p in papers:
                pid = p.get("pmid") or p.get("paper_id") or ""
                if pid:
                    pid_to_paper[str(pid)] = p
    for p in (local_paper_metadata or []):
        pid = p.get("paper_id") or ""
        if pid:
            pid_to_paper[str(pid)] = p

    buffer = BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
    )
    S     = _styles()
    story: list = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("Neurohypothesis — Neuroscience Hypothesis Report", S["title"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M UTC')}",
        S["subtitle"],
    ))
    story.append(_hr(thick=1.0))

    # ── Research topic ────────────────────────────────────────────────────
    story.append(Paragraph("Research Topic", S["section"]))
    story.append(Paragraph(topic, S["body"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"{len(hypotheses)} hypothesis/hypotheses generated in this session.",
        S["caption"],
    ))

    # ── One section per hypothesis ────────────────────────────────────────
    for idx, hyp in enumerate(hypotheses):
        story.extend(_build_hypothesis_section(
            hyp, idx, S, doc.width,
            pid_to_paper=pid_to_paper,
        ))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.8 * cm))
    story.append(_hr())
    story.append(Paragraph(config.DISCLAIMER, S["caption"]))
    story.append(Paragraph("Generated by Neurohypothesis", S["caption"]))

    doc.build(story)
    return buffer.getvalue()


def export_session_to_pdf(
    topic:      str,
    hypotheses: list[dict[str, Any]],
    session_id: str,
    cat_papers: dict[str, list[dict]] | None = None,
) -> Path:
    """
    Write the session PDF to config.EXPORTS_DIR and return the file path.

    Called by N19 (export_results).  The file is also accessible via
    st.session_state["export_path"] for the download button in app.py.

    Args:
        topic:      user's research topic (used in filename + report header).
        hypotheses: list of Hypothesis dicts from AgentState.
        session_id: used to create a unique filename per session.
        cat_papers: optional category->papers dict for APA-style citations.

    Returns:
        Path to the written PDF file.
    """
    config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build a filesystem-safe filename from the topic
    safe_topic = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else ""
        for c in topic[:40]
    ).strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"Neurohypothesis_{safe_topic}_{timestamp}.pdf"
    out_path  = config.EXPORTS_DIR / filename

    try:
        pdf_bytes = render_pdf_bytes(topic, hypotheses, cat_papers=cat_papers)
        out_path.write_bytes(pdf_bytes)
        logger.info(f"[N19] PDF written: {out_path} ({len(pdf_bytes):,} bytes)")
    except Exception as exc:
        logger.error(f"[N19] PDF export failed: {exc}")
        raise

    return out_path
