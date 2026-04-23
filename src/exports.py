"""
PDF export for the Hypothesis Generator.

Generates a clean PDF report containing:
    - User query
    - Gap analysis table (past summary, future summary, gap score)
    - Generated hypothesis + originality grade
    - PubMed check results (if run)
    - Supporting papers
    - Suggested approach

Uses reportlab for PDF generation (pure Python, no system dependencies).
Install: pip install reportlab

Public API:
    - export_to_pdf(query, gap_results) -> bytes
      Returns the PDF as bytes for Streamlit download.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# =============================================================================
# Colour constants (matching app palette — blue tones)
# =============================================================================

BLUE_DARK = colors.HexColor("#1a4f8a")
BLUE_MED = colors.HexColor("#2e7bc4")
BLUE_LIGHT = colors.HexColor("#7ab3e0")
GREY_LIGHT = colors.HexColor("#f5f5f5")
GREY_BORDER = colors.HexColor("#dddddd")
BLACK = colors.black
WHITE = colors.white


# =============================================================================
# Style helpers
# =============================================================================

def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            fontSize=18,
            fontName="Helvetica-Bold",
            textColor=BLUE_DARK,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontSize=10,
            fontName="Helvetica",
            textColor=colors.HexColor("#555555"),
            spaceAfter=12,
        ),
        "section": ParagraphStyle(
            "section",
            fontSize=13,
            fontName="Helvetica-Bold",
            textColor=BLUE_DARK,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontSize=10,
            fontName="Helvetica",
            textColor=BLACK,
            spaceAfter=4,
            leading=14,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontSize=10,
            fontName="Helvetica",
            textColor=BLACK,
            spaceAfter=3,
            leftIndent=12,
            leading=14,
        ),
        "hypothesis": ParagraphStyle(
            "hypothesis",
            fontSize=11,
            fontName="Helvetica-Bold",
            textColor=BLACK,
            spaceAfter=6,
            leading=16,
        ),
        "grade": ParagraphStyle(
            "grade",
            fontSize=10,
            fontName="Helvetica",
            textColor=BLUE_MED,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontSize=9,
            fontName="Helvetica-Oblique",
            textColor=colors.HexColor("#777777"),
            spaceAfter=4,
        ),
    }


def _hr():
    return HRFlowable(
        width="100%", thickness=0.5,
        color=GREY_BORDER, spaceAfter=6, spaceBefore=6,
    )


# =============================================================================
# Grade label helper
# =============================================================================

def _grade_label(score: float | None, label: str) -> str:
    if score is None:
        return label
    return f"{label} (score {score:.2f})"


# =============================================================================
# Main export function
# =============================================================================

def export_to_pdf(
    query: str,
    gap_results: dict[str, Any],
) -> bytes:
    """
    Generate a PDF report from the session results.

    Args:
        query: the user's original research query.
        gap_results: the dict stored in st.session_state.gap_results,
                     containing past_summary, future_summary,
                     literature_gap, and hypotheses.

    Returns:
        PDF content as bytes, ready for st.download_button().
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    S = _styles()
    story = []

    # --- Header ---
    story.append(Paragraph("Hypothesis Generator — Research Report", S["title"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        S["subtitle"],
    ))
    story.append(_hr())

    # --- Research query ---
    story.append(Paragraph("Research Query", S["section"]))
    story.append(Paragraph(query, S["body"]))
    story.append(Spacer(1, 0.3 * cm))

    # --- Gap Analysis table ---
    past = gap_results.get("past_summary", [])
    future = gap_results.get("future_summary", [])
    lit_gap = gap_results.get("literature_gap", {})

    if past or future:
        story.append(Paragraph("Gap Analysis", S["section"]))

        # Literature gap score line
        if lit_gap:
            gap_label = lit_gap.get("label", "")
            gap_score = lit_gap.get("score", 0.0)
            gap_sim = lit_gap.get("similarity", 0.0)
            story.append(Paragraph(
                f"Literature gap: {gap_label} — gap score {gap_score:.2f}, similarity {gap_sim:.2f}",
                S["grade"],
            ))
            story.append(Paragraph(
                "Higher gap score = past findings and future recommendations differ more.",
                S["caption"],
            ))

        # Two-column table
        max_rows = max(len(past), len(future), 1)
        past_pad = past + [""] * (max_rows - len(past))
        future_pad = future + [""] * (max_rows - len(future))

        table_data = [
            [
                Paragraph("<b>Past — What Has Been Done</b>", S["body"]),
                Paragraph("<b>Future — What Is Recommended</b>", S["body"]),
            ]
        ]
        for p, f in zip(past_pad, future_pad):
            table_data.append([
                Paragraph(p, S["bullet"]),
                Paragraph(f, S["bullet"]),
            ])

        col_w = (doc.width - 0.5 * cm) / 2
        tbl = Table(table_data, colWidths=[col_w, col_w])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GREY_LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.4, GREY_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.3 * cm))

    # --- Generated hypothesis ---
    hypotheses = gap_results.get("hypotheses", [])
    if hypotheses:
        h = hypotheses[0]
        story.append(_hr())
        story.append(Paragraph("Generated Hypothesis", S["section"]))
        story.append(Paragraph(h.get("statement", ""), S["hypothesis"]))

        # Originality grade
        orig_score = h.get("originality_score")
        orig_label = h.get("grade_label", "")
        orig_sim = h.get("max_similarity_to_past", 0.0)
        if orig_label:
            story.append(Paragraph(
                f"Originality (vs local library): {orig_label} "
                f"(score {orig_score:.2f}, similarity {orig_sim:.2f})",
                S["grade"],
            ))

        # Supporting papers
        papers = h.get("supported_by", [])
        if papers:
            story.append(Paragraph("Supporting papers:", S["body"]))
            for pid in papers:
                story.append(Paragraph(f"• {pid}", S["bullet"]))

        # Suggested approach
        approach = h.get("suggested_approach", [])
        if isinstance(approach, str):
            approach = [approach] if approach.strip() else []
        if approach:
            story.append(Paragraph("Suggested approach:", S["body"]))
            for step in approach:
                story.append(Paragraph(f"• {step}", S["bullet"]))

        # PubMed check (if run)
        pubmed = h.get("pubmed", {})
        if not pubmed:
            # Also check session_state key pattern (passed in from app.py if needed)
            pass

        if pubmed:
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph("PubMed Check (last 5 years)", S["section"]))
            pm_status = pubmed.get("status", "")
            pm_query = pubmed.get("query_used", "")
            if pm_status == "ok":
                pm_label = pubmed.get("label", "")
                pm_sim = pubmed.get("max_similarity", 0.0)
                pm_total = pubmed.get("total_found", 0)
                pm_compared = pubmed.get("papers_compared", 0)
                story.append(Paragraph(
                    f"{pm_label} (similarity {pm_sim:.2f})",
                    S["grade"],
                ))
                story.append(Paragraph(
                    f"Found {pm_total} papers for \"{pm_query}\" · compared top {pm_compared}",
                    S["caption"],
                ))
                matches = pubmed.get("matches", [])
                if matches:
                    story.append(Paragraph("Matching recent papers:", S["body"]))
                    for m in matches:
                        title = m.get("title", "")
                        year = m.get("year", "")
                        sim = m.get("similarity", 0.0)
                        year_str = f" ({year})" if year else ""
                        story.append(Paragraph(
                            f"• {title}{year_str} — similarity {sim:.2f}",
                            S["bullet"],
                        ))
            elif pm_status == "no_results":
                story.append(Paragraph(
                    f"No recent matches found for \"{pm_query}\".",
                    S["body"],
                ))
            elif pm_status == "error":
                story.append(Paragraph("PubMed was not available.", S["caption"]))
        else:
            story.append(Paragraph(
                "PubMed check: not run in this session.",
                S["caption"],
            ))

    # --- Footer ---
    story.append(Spacer(1, 0.5 * cm))
    story.append(_hr())
    story.append(Paragraph(
        "Generated by Hypothesis Generator · For research purposes only.",
        S["caption"],
    ))

    doc.build(story)
    return buffer.getvalue()
