# exporter.py
# Generates a clean self-contained HTML session report.

import re
import html as html_lib
from datetime import datetime


def build_html_report(
    role: str,
    interview_type: str,
    difficulty: str,
    technique: str,
    messages: list,
    scores: list,
    jd_summary: str = "",
    cv_summary: str = "",
    gap_analysis: str = "",
) -> str:
    date_str = datetime.now().strftime("%B %d, %Y — %H:%M")
    avg_score = sum(scores) / len(scores) if scores else 0

    if avg_score >= 8:
        perf_msg = "🌟 Outstanding performance! You're well-prepared for this interview."
        perf_color = "#005a8e"
    elif avg_score >= 6:
        perf_msg = "✅ Good work! A bit more practice on your weaker areas and you'll be ready."
        perf_color = "#007a59"
    elif avg_score >= 4:
        perf_msg = "📈 Decent effort — review the improved answers and try again."
        perf_color = "#7a5000"
    else:
        perf_msg = "💪 Keep practising! Study the model answers and focus on structure."
        perf_color = "#a03000"

    qa_blocks = _build_qa_blocks(messages, scores)
    score_bars = _build_score_bars(scores)

    gap_section = ""
    if jd_summary or gap_analysis:
        content = gap_analysis if gap_analysis else jd_summary
        title = "🔎 Gap Analysis" if gap_analysis else "📄 Job Description Analysis"
        gap_section = f"""
        <section class="section">
            <h2>{title}</h2>
            <div class="analysis-box">{_md_to_html(html_lib.escape(content))}</div>
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interview Session Report — {role or "Candidate"}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'DM Sans', sans-serif;
            background: #f7f8fc;
            color: #1a1a2e;
            line-height: 1.7;
        }}
        .page {{ max-width: 860px; margin: 0 auto; padding: 48px 32px; }}

        /* Header */
        .header {{ border-bottom: 2px solid #d4d8e8; padding-bottom: 32px; margin-bottom: 40px; }}
        .header h1 {{
            font-family: 'Syne', sans-serif; font-size: 2.2rem;
            font-weight: 800; color: #1a1a2e; margin-bottom: 8px;
        }}
        .header .subtitle {{ color: #555; font-size: 0.95rem; margin-bottom: 20px; }}
        .meta-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px; margin-top: 20px;
        }}
        .meta-item {{
            background: #ffffff; border: 1px solid #d4d8e8;
            border-radius: 8px; padding: 12px 16px;
        }}
        .meta-item .label {{
            font-size: 0.72rem; color: #555; text-transform: uppercase;
            letter-spacing: 0.08em; margin-bottom: 4px;
        }}
        .meta-item .value {{ font-size: 0.95rem; color: #1a1a2e; font-weight: 500; }}

        /* Sections */
        .section {{ margin-bottom: 48px; }}
        .section h2 {{
            font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700;
            color: #1a1a2e; margin-bottom: 20px; padding-bottom: 8px;
            border-bottom: 1px solid #d4d8e8;
        }}

        /* Performance */
        .performance-banner {{
            background: #ffffff; border-left: 4px solid {perf_color};
            border-radius: 8px; padding: 16px 20px; margin-bottom: 24px;
            font-size: 1rem; color: {perf_color}; font-weight: 500;
            border: 1px solid #d4d8e8;
        }}

        /* Score summary cards */
        .score-summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 28px; }}
        .score-card {{
            background: #ffffff; border: 1px solid #d4d8e8;
            border-radius: 10px; padding: 20px; text-align: center;
        }}
        .score-card .number {{
            font-family: 'Syne', sans-serif; font-size: 2rem;
            font-weight: 800; color: #3a9fd4;
        }}
        .score-card .label {{ font-size: 0.8rem; color: #555; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }}

        /* Score bars */
        .score-bar-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; font-size: 0.88rem; }}
        .score-bar-label {{ width: 32px; color: #555; flex-shrink: 0; }}
        .score-bar-track {{ flex: 1; background: #e8ecf4; border-radius: 4px; height: 10px; overflow: hidden; }}
        .score-bar-fill {{ height: 100%; border-radius: 4px; }}
        .score-bar-value {{ width: 40px; text-align: right; color: #555; flex-shrink: 0; }}

        /* Analysis box */
        .analysis-box {{
            background: #ffffff; border: 1px solid #d4d8e8;
            border-radius: 10px; padding: 20px 24px; font-size: 0.93rem; color: #1a1a2e;
        }}
        .analysis-box strong {{ color: #1a1a2e; }}
        .analysis-box ul {{ padding-left: 20px; margin: 6px 0; }}
        .analysis-box li {{ margin-bottom: 4px; color: #333; }}
        .analysis-box p {{ margin-bottom: 8px; color: #333; }}

        /* Q&A blocks */
        .qa-block {{ margin-bottom: 32px; border: 1px solid #d4d8e8; border-radius: 12px; overflow: hidden; }}
        .qa-header {{
            background: #eef4fb; border-bottom: 1px solid #d4d8e8;
            padding: 14px 20px; display: flex; justify-content: space-between; align-items: center;
        }}
        .qa-header .q-label {{
            font-family: 'Syne', sans-serif; font-size: 0.85rem; font-weight: 700;
            color: #0072B2; text-transform: uppercase; letter-spacing: 0.06em;
        }}
        .score-badge {{ font-size: 0.82rem; font-weight: 600; padding: 3px 12px; border-radius: 20px; background: #f7f8fc; }}
        .question-text {{
            padding: 16px 20px; font-size: 1rem; color: #1a1a2e;
            background: #ffffff; border-bottom: 1px solid #d4d8e8;
        }}

        /* Answer attempts */
        .attempt-block {{ border-bottom: 1px solid #d4d8e8; }}
        .attempt-block:last-child {{ border-bottom: none; }}
        .answer-section {{
            padding: 14px 20px; background: #e8f0fe;
            border-left: 3px solid #3a9fd4; border-bottom: 1px solid #d4d8e8;
        }}
        .answer-section.improved {{
            background: #dce8fd;
            border-left: 3px solid #2a7fc0; border-bottom: 1px solid #d4d8e8;
        }}
        .block-label {{
            font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
            margin-bottom: 6px; font-weight: 600; color: #2a7fc0;
        }}
        .block-label.improved {{ color: #1a5fa0; }}
        .block-content {{ font-size: 0.93rem; color: #333; }}
        .feedback-section {{
            padding: 16px 20px; background: #f5fbf8; border-bottom: 1px solid #d4d8e8;
        }}
        .feedback-label {{
            font-size: 0.75rem; color: #007a59; text-transform: uppercase;
            letter-spacing: 0.08em; margin-bottom: 10px; font-weight: 600;
        }}
        .feedback-section strong {{ color: #1a1a2e; }}
        .feedback-section ul {{ padding-left: 18px; margin: 4px 0 10px; }}
        .feedback-section li {{ color: #333; margin-bottom: 3px; font-size: 0.91rem; }}
        .feedback-section p {{ margin-bottom: 8px; font-size: 0.91rem; color: #333; }}
        .feedback-section h4 {{ color: #1a1a2e; margin: 8px 0 4px; font-size: 0.95rem; }}

        /* Footer */
        .footer {{
            margin-top: 60px; padding-top: 24px; border-top: 1px solid #d4d8e8;
            text-align: center; color: #555; font-size: 0.8rem;
        }}

        @media print {{
            body {{ background: white; }}
            .page {{ padding: 24px; }}
        }}
    </style>
</head>
<body>
<div class="page">

    <div class="header">
        <h1>🎯 Interview Session Report</h1>
        <p class="subtitle">Generated on {date_str}</p>
        <div class="meta-grid">
            <div class="meta-item"><div class="label">Role</div><div class="value">{role or "Not specified"}</div></div>
            <div class="meta-item"><div class="label">Interview Type</div><div class="value">{interview_type}</div></div>
            <div class="meta-item"><div class="label">Difficulty</div><div class="value">{difficulty}</div></div>
            <div class="meta-item"><div class="label">Prompt Technique</div><div class="value">{technique}</div></div>
            <div class="meta-item"><div class="label">Questions Completed</div><div class="value">{len(scores)}</div></div>
            <div class="meta-item"><div class="label">Average Score</div><div class="value">{avg_score:.1f} / 10</div></div>
        </div>
    </div>

    <section class="section">
        <h2>📊 Performance Summary</h2>
        <div class="performance-banner">{perf_msg}</div>
        <div class="score-summary">
            <div class="score-card"><div class="number">{avg_score:.1f}</div><div class="label">Average Score</div></div>
            <div class="score-card"><div class="number">{max(scores) if scores else 0}</div><div class="label">Best Answer</div></div>
            <div class="score-card"><div class="number">{min(scores) if scores else 0}</div><div class="label">Lowest Answer</div></div>
        </div>
        {score_bars}
    </section>

    {gap_section}

    <section class="section">
        <h2>💬 Questions & Feedback</h2>
        {qa_blocks}
    </section>

    <div class="footer">Generated by AI Interview Coach &nbsp;·&nbsp; Turing College AI Engineering Course</div>

</div>
</body>
</html>"""
    return html


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def _build_score_bars(scores: list) -> str:
    if not scores:
        return ""
    bars = ""
    for i, s in enumerate(scores, 1):
        pct = s * 10
        color = "#3a9fd4" if s >= 7 else "#56B4E9" if s >= 5 else "#D55E00"
        bars += f"""
        <div class="score-bar-row">
            <span class="score-bar-label">Q{i}</span>
            <div class="score-bar-track">
                <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <span class="score-bar-value">{s}/10</span>
        </div>"""
    return bars


def _extract_feedback(text: str) -> str:
    m = re.search(r"---FEEDBACK---(.*?)---END FEEDBACK---", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Strip any next question block and return the rest as feedback
    cleaned = re.sub(r"---NEXT QUESTION---.*?---END QUESTION---", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _extract_question(text: str) -> str:
    m = re.search(r"---NEXT QUESTION---(.*?)---END QUESTION---", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _build_qa_blocks(messages: list, scores: list) -> str:
    """
    Parse message history into Q&A blocks, grouping all attempts per question.
    Each block shows the question, then all answer+feedback pairs in order.
    """
    INTERNAL = {
        "I am ready. Please ask me the first question.",
        "Please skip this question and move to the next one.",
        "I am satisfied with my answer. Please move to the next question.",
    }

    # Build structured list: [{question, attempts:[{answer,feedback}]}]
    questions = []
    current_q = ""
    current_attempts = []  # list of {answer, feedback}

    for msg in messages:
        if msg["role"] == "assistant":
            fb = _extract_feedback(msg["content"])
            nq = _extract_question(msg["content"])

            # Opening message: first assistant message contains the first question
            if not current_q and not fb.strip():
                # This is the opening question only message
                current_q = nq or msg["content"]
                current_attempts = []

            elif fb:
                # Attach feedback to the most recent answer attempt
                if current_attempts:
                    current_attempts[-1]["feedback"] = fb

                # If AI also included the next question, save current and start new
                if nq:
                    questions.append({"question": current_q, "attempts": current_attempts})
                    current_q = nq
                    current_attempts = []

        elif msg["role"] == "user" and msg["content"] not in INTERNAL:
            current_attempts.append({"answer": msg["content"], "feedback": ""})

    # Save the last question
    if current_q:
        questions.append({"question": current_q, "attempts": current_attempts})

    if not questions:
        return "<p style='color:#555;'>No question data available.</p>"

    blocks = ""
    for i, q_data in enumerate(questions, 1):
        score = scores[i - 1] if i - 1 < len(scores) else None
        blocks += _render_qa_block(i, q_data["question"], q_data["attempts"], score)
    return blocks


def _render_qa_block(number: int, question: str, attempts: list, score: int | None) -> str:
    if score is not None:
        sc = "#3a9fd4" if score >= 7 else "#56B4E9" if score >= 5 else "#D55E00"
        score_html = f'<span class="score-badge" style="color:{sc};border:1px solid {sc};">{score}/10</span>'
    else:
        score_html = ""

    q_html = f'<div class="question-text">{_md_to_html(html_lib.escape(question))}</div>' if question else ""

    attempts_html = ""
    for idx, att in enumerate(attempts):
        is_improved = idx > 0
        label = "Your Answer" if not is_improved else f"Your Improved Answer (Attempt {idx + 1})"
        label_class = "block-label improved" if is_improved else "block-label"
        answer_class = "answer-section improved" if is_improved else "answer-section"

        if att.get("answer"):
            attempts_html += (
                f'<div class="attempt-block">'
                f'<div class="{answer_class}">'
                f'<div class="{label_class}">{label}</div>'
                f'<div class="block-content">{_md_to_html(html_lib.escape(att["answer"]))}</div>'
                f'</div>'
            )
            if att.get("feedback"):
                attempts_html += (
                    f'<div class="feedback-section">'
                    f'<div class="feedback-label">AI Feedback</div>'
                    f'{_md_to_html(html_lib.escape(att["feedback"]))}'
                    f'</div>'
                )
            attempts_html += '</div>'

    return f"""
    <div class="qa-block">
        <div class="qa-header">
            <span class="q-label">Question {number}</span>
            {score_html}
        </div>
        {q_html}
        {attempts_html}
    </div>"""


def _md_to_html(text: str) -> str:
    """Convert simple markdown to HTML."""
    if not text:
        return ""
    lines = text.split("\n")
    html_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", stripped[2:])
            html_lines.append(f"<li>{content}</li>")
            continue

        if in_list:
            html_lines.append("</ul>")
            in_list = False

        stripped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", stripped)

        if stripped.startswith("### "):
            html_lines.append(f"<h4>{stripped[4:]}</h4>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h3>{stripped[3:]}</h3>")
        elif stripped.startswith("# "):
            html_lines.append(f"<h3>{stripped[2:]}</h3>")
        else:
            html_lines.append(f"<p>{stripped}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)
