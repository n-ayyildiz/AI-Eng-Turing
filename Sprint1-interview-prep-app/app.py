# app.py — AI Interview Coach
# Single-page interview preparation app built with Streamlit + OpenAI.

import os
import re
import html as html_lib
import io
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

from prompts import (
    PROMPT_TECHNIQUES,
    TECHNIQUE_EXPLANATIONS,
    PERSONAS,
    build_system_prompt,
)
from security import validate_input, check_job_description, check_cv_file
from pricing import calculate_cost, format_session_cost
from cv_parser import extract_cv_text, summarise_cv, analyse_job_description, analyse_gap
from exporter import build_html_report

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None,
    }
)

# ----------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #1a1a2e; }
    h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; color: #1a1a2e !important; }
    .stApp { background-color: #f7f8fc; color: #1a1a2e; }

    [data-testid="stSidebar"] { background-color: #eef0f7; border-right: 1px solid #d4d8e8; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: #1a1a2e !important; }

    [data-testid="stSlider"] > div > div > div > div { background: #3a9fd4 !important; }
    [data-testid="stSlider"] > div > div > div { background: #c2e0f2 !important; }

    .chat-user {
        background: #e8f0fe; border-left: 4px solid #3a9fd4;
        padding: 12px 16px; border-radius: 8px; margin: 8px 0; color: #1a1a2e;
    }
    .feedback-box {
        background: #f0faf7; border: 1px solid #b3ddd4;
        border-left: 4px solid #009E73; padding: 16px 20px;
        border-radius: 10px; margin: 10px 0; color: #1a1a2e;
    }
    .question-box {
        background: #eef4fb; border: 1px solid #b3cfe8;
        border-left: 4px solid #3a9fd4; padding: 16px 20px;
        border-radius: 10px; margin: 10px 0; color: #1a1a2e;
    }
    .nudge-box {
        background: #fdf3e3; border-left: 4px solid #E69F00;
        padding: 10px 16px; border-radius: 8px;
        margin: 8px 0; font-size: 0.9rem; color: #7a5000;
    }
    .summary-box {
        background: #ffffff; border: 1px solid #d4d8e8;
        border-radius: 12px; padding: 20px 24px; margin: 10px 0; color: #1a1a2e;
    }
    .stTextArea textarea {
        background-color: #ffffff !important; color: #1a1a2e !important;
        border: 1px solid #c4c8d8 !important; border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextArea label { color: #1a1a2e !important; }
    .stTextInput input {
        background-color: #ffffff !important; color: #1a1a2e !important;
        border: 1px solid #c4c8d8 !important; border-radius: 8px !important;
    }
    .stTextInput label { color: #1a1a2e !important; }
    .stSelectbox > div > div {
        background-color: #ffffff !important; border: 1px solid #c4c8d8 !important;
        color: #1a1a2e !important; border-radius: 8px !important;
    }
    .stSelectbox label { color: #1a1a2e !important; }
    .stButton > button {
        background: #3a9fd4; color: #ffffff; border: none;
        border-radius: 8px; font-family: 'Syne', sans-serif;
        font-weight: 600; padding: 0.5rem 1.5rem; transition: background 0.2s;
    }
    .stButton > button:hover { background: #2a8bbf; color: #ffffff; }
    .stDownloadButton > button {
        background: #009E73; color: #ffffff; border: none;
        border-radius: 8px; font-family: 'Syne', sans-serif;
        font-weight: 600; padding: 0.5rem 1.5rem; transition: background 0.2s;
    }
    .stDownloadButton > button:hover { background: #007a59; color: #ffffff; }
    [data-testid="stFileUploader"] {
        background: #ffffff; border: 1px dashed #b3cfe8; border-radius: 8px; padding: 8px;
    }
    [data-testid="stFileUploader"] label { color: #1a1a2e !important; }
    .streamlit-expanderHeader { background: #eef0f7 !important; color: #1a1a2e !important; border-radius: 8px !important; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #d4d8e8; border-radius: 10px; padding: 16px; }
    [data-testid="stMetricLabel"] { color: #555 !important; }
    [data-testid="stMetricValue"] { color: #3a9fd4 !important; }
    .stProgress > div > div > div { background: #3a9fd4 !important; }
    hr { border-color: #d4d8e8 !important; }
    .stCaption, caption { color: #666 !important; }
    .cost-badge { font-size: 0.78rem; color: #888; margin-top: 4px; }
    .stAlert { border-radius: 8px !important; }
    /* Hide the Deploy button from the toolbar */
    [data-testid="stToolbar"] { visibility: hidden; }
    [data-testid="stToolbar"]::before { visibility: visible; }

    /* Hide the sidebar collapse arrow so it can never be hidden */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    button[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
def init_state():
    defaults = {
        "messages": [],
        "interview_started": False,
        "interview_done": False,
        "question_count": 1,
        "total_questions": 4,
        "attempt": 0,
        "awaiting_decision": False,  # True after feedback, waiting for user to choose next action
        "is_final_answer": False,    # True only when user answered the actual last question
        "scores": [],
        "cost_log": [],
        "last_cost": None,
        "jd_summary": "",
        "cv_summary": "",
        "gap_analysis": "",
        "jd_text": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings for Interview Coach")
    st.markdown("### 🧠 Prompt Technique")
    technique_label = st.selectbox("Select Technique", list(PROMPT_TECHNIQUES.keys()))
    st.caption(TECHNIQUE_EXPLANATIONS[technique_label])
    persona_key = "Neutral 😐"
    if technique_label == "Persona-Based":
        persona_key = st.selectbox("Interviewer Persona", list(PERSONAS.keys()), index=1,
            help="Friendly = encouraging, Neutral = balanced, Strict = demanding.")
    st.markdown("### 🤖 Model & Parameters")
    model = st.selectbox("Model",
        ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1", "gpt-4o", "gpt-4o-mini"], index=0,
        help="gpt-4.1-mini offers strong quality at low cost.")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.4, 0.05,
        help="Low = consistent scoring. Recommended: 0.4.")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05,
        help="Controls vocabulary diversity. Recommended: 0.9.")
    max_tokens = st.slider("Max Response Tokens", 200, 1500, 700, 50,
        help="Higher = more detailed feedback but slower.")
    frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.3, 0.1,
        help="Reduces repetition. Recommended: 0.3.")
    st.markdown("### 💰 Session Cost")
    if st.session_state["cost_log"]:
        st.markdown(format_session_cost(st.session_state["cost_log"]))
    else:
        st.caption("Cost will appear after the first API call.")
    st.markdown("---")
    if st.button("🔄 Reset Everything", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Interview setup variables — read from session state after interview starts
# Before start: set by widgets on the main page
# After start: persisted in session state
role = st.session_state.get("role", "")
interview_type = st.session_state.get("interview_type", "Behavioral")
difficulty = st.session_state.get("difficulty", "Medium")
total_questions = st.session_state.get("total_questions", 4)

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def call_openai(system_prompt: str, messages: list) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )
    usage = response.usage
    cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)
    st.session_state["cost_log"].append(cost)
    st.session_state["last_cost"] = cost["formatted"]
    return response.choices[0].message.content


def get_system_prompt(is_last_question: bool = False) -> str:
    prompt = build_system_prompt(
        technique_key=PROMPT_TECHNIQUES[technique_label],
        role=role,
        interview_type=interview_type,
        difficulty=difficulty,
        persona_key=persona_key,
        jd_summary=st.session_state["jd_summary"],
        cv_summary=st.session_state["cv_summary"],
        gap_analysis=st.session_state.get("gap_analysis", ""),
    )
    if is_last_question:
        prompt += (
            "\n\nIMPORTANT: This is the LAST question of the session. "
            "After giving feedback on the candidate's answer, do NOT ask another question. "
            "End your response with just the feedback block. No ---NEXT QUESTION--- block."
        )
    return prompt


def extract_score(text: str) -> int | None:
    match = re.search(r"\*\*Score:\*\*\s*(\d+)\s*/\s*10", text)
    return int(match.group(1)) if match else None


def split_feedback_and_question(text: str) -> tuple[str, str]:
    feedback, question = "", ""
    fb_match = re.search(r"---FEEDBACK---(.*?)---END FEEDBACK---", text, re.DOTALL)
    q_match = re.search(r"---NEXT QUESTION---(.*?)---END QUESTION---", text, re.DOTALL)
    if fb_match:
        feedback = fb_match.group(1).strip()
    if q_match:
        question = q_match.group(1).strip()
    if not feedback and not question:
        feedback = text
    return feedback, question


INTERNAL_MSGS = {
    "I am ready. Please ask me the first question.",
    "Please skip this question and move to the next one.",
    "I am satisfied with my answer. Please move to the next question.",
    "*(skipped)*",
}


def render_chat_history():
    """
    Render all past messages with correct sequential question numbering.
    When awaiting_decision is True, the next question from the last assistant
    message is suppressed — it appears only after the user clicks Next Question.
    This ensures the nudge and action buttons always appear BEFORE the next question.
    """
    q_counter = 0
    shown_opening = False
    messages = st.session_state["messages"]
    awaiting = st.session_state.get("awaiting_decision", False)

    # Find index of the last assistant message so we can suppress its next question
    last_assistant_idx = -1
    if awaiting:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                last_assistant_idx = i
                break

    for idx, msg in enumerate(messages):
        if msg["role"] == "assistant" and not shown_opening:
            shown_opening = True
            q_counter = 1
            _, q = split_feedback_and_question(msg["content"])
            st.markdown(
                f"<div class='question-box'>🤖 <strong>Question 1:</strong><br>"
                f"{html_lib.escape(q or msg['content'])}</div>",
                unsafe_allow_html=True,
            )

        elif msg["role"] == "user" and msg["content"] not in INTERNAL_MSGS:
            st.markdown(
                f"<div class='chat-user'>👤 <strong>You:</strong><br>{html_lib.escape(msg['content'])}</div>",
                unsafe_allow_html=True,
            )

        elif msg["role"] == "assistant" and shown_opening:
            fb, q = split_feedback_and_question(msg["content"])
            if fb:
                st.markdown(
                    f"<div class='feedback-box'>📊 <strong>Feedback:</strong><br>{html_lib.escape(fb)}</div>",
                    unsafe_allow_html=True,
                )
            # Only show next question if this is NOT the last assistant message
            # while awaiting a decision — prevents question appearing before nudge/buttons
            if q and not (awaiting and idx == last_assistant_idx):
                q_counter += 1
                st.markdown(
                    f"<div class='question-box'>🤖 <strong>Question {q_counter}:</strong><br>{html_lib.escape(q)}</div>",
                    unsafe_allow_html=True,
                )


# ----------------------------------------------------------------------
# PAGE HEADER
# ----------------------------------------------------------------------
st.markdown(
    "<h1 style='font-family:Syne,sans-serif;font-size:2.2rem;margin-bottom:0;color:#1a1a2e;'>"
    "🎯 AI Interview Coach</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#555;margin-top:4px;font-size:0.95rem;'>"
    "Practice interviews with AI-powered feedback & suggestions — scored, structured, and practical.</p>",
    unsafe_allow_html=True,
)

# ======================================================================
# SECTION 1 — SETUP
# ======================================================================
if not st.session_state["interview_started"]:
    # --- Row 1: Job Role, Interview Type, Difficulty, No. of Questions ---
    col1, col2, col3, col4 = st.columns([3, 3, 3, 1.5])
    with col1:
        role = st.text_input("🧑‍💼 Job Role",
            placeholder="e.g. Data Scientist",
            help="The job title you are preparing to interview for.")
    with col2:
        interview_type = st.selectbox("🎙️ Interview Type",
            ["Behavioral", "Technical", "Mixed (Behavioral + Technical)"],
            help="Behavioral = soft skills. Technical = domain knowledge.")
    with col3:
        difficulty = st.selectbox("📊 Difficulty",
            ["Easy", "Medium", "Hard"], index=1,
            help="Easy = entry-level, Medium = mid-level, Hard = senior/lead.")
    with col4:
        total_questions = st.selectbox("🔢 Questions",
            options=list(range(1, 11)),
            index=3,
            help="How many questions in this session (1–10).")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📄 Job Description** *(optional)*")
        jd_input = st.text_area("jd", placeholder="Paste the job posting here...",
            height=160, label_visibility="collapsed")
    with col2:
        st.markdown("**📁 Your CV** *(optional)*")
        cv_file = st.file_uploader("cv", type=["pdf", "docx"], label_visibility="collapsed")
        st.caption("PDF or Word (.docx) — max 5 MB")

    st.markdown(
        "<div style='background:#fff8e1;border-left:4px solid #E69F00;"
        "border-radius:8px;padding:10px 16px;margin:8px 0;font-size:0.82rem;color:#7a5000;'>"
        "🔒 <strong>Privacy Notice:</strong> Any job description or CV content you provide "
        "will be sent to the OpenAI API for analysis. Please do not upload documents containing "
        "sensitive personal information. This app is a demo project and is not intended for "
        "production use. By clicking Start Interview you acknowledge this."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("🚀 Start Interview"):
        if not role.strip():
            st.warning("⚠️ Please enter a job role before starting.")
        else:
            # Validate inputs
            if jd_input.strip():
                jd_ok, jd_err = check_job_description(jd_input)
                if not jd_ok:
                    st.error(jd_err)
                    st.stop()
            if cv_file:
                cv_ok, cv_err = check_cv_file(cv_file)
                if not cv_ok:
                    st.error(cv_err)
                    st.stop()

            # Persist setup values to session state
            st.session_state["role"] = role
            st.session_state["interview_type"] = interview_type
            st.session_state["difficulty"] = difficulty
            st.session_state["total_questions"] = total_questions
            st.session_state["jd_text"] = jd_input.strip()

            if cv_file:
                st.session_state["_cv_bytes"] = cv_file.getvalue()
                st.session_state["_cv_ext"] = cv_file.name.rsplit(".", 1)[-1].lower()
            else:
                st.session_state["_cv_bytes"] = None

            # Silent document analysis
            with st.spinner("⏳ Preparing your personalised interview session..."):
                jd_summary, cv_summary, gap_analysis = "", "", ""
                if st.session_state["jd_text"]:
                    ok, jd_summary = analyse_job_description(st.session_state["jd_text"], model)
                    if not ok:
                        st.error(jd_summary); st.stop()
                if st.session_state.get("_cv_bytes"):
                    file_obj = io.BytesIO(st.session_state["_cv_bytes"])
                    file_obj.name = f"cv.{st.session_state['_cv_ext']}"
                    ok, cv_text = extract_cv_text(file_obj)
                    if not ok:
                        st.error(cv_text); st.stop()
                    ok, cv_summary = summarise_cv(cv_text, model)
                    if not ok:
                        st.error(cv_summary); st.stop()
                if jd_summary and cv_summary:
                    ok, gap_analysis = analyse_gap(cv_summary, jd_summary, model)
                    if not ok:
                        gap_analysis = ""
                st.session_state["jd_summary"] = jd_summary
                st.session_state["cv_summary"] = cv_summary
                st.session_state["gap_analysis"] = gap_analysis

            # Get first question
            with st.spinner("Preparing your first question..."):
                opening = call_openai(
                    get_system_prompt(),
                    [{"role": "user", "content": "I am ready. Please ask me the first question."}],
                )
                st.session_state["messages"] = [
                    {"role": "user", "content": "I am ready. Please ask me the first question."},
                    {"role": "assistant", "content": opening},
                ]
                st.session_state["question_count"] = 1
                st.session_state["attempt"] = 0
                st.session_state["interview_started"] = True
            st.rerun()

# ======================================================================
# SECTION 2 — INTERVIEW
# ======================================================================
if st.session_state["interview_started"] and not st.session_state["interview_done"]:

    render_chat_history()

    q_num = st.session_state["question_count"]
    total_q = st.session_state["total_questions"]
    attempt = st.session_state["attempt"]

    st.markdown("---")

    # Progress bar
    progress = max(0.0, min((q_num - 1) / total_q, 1.0))
    st.progress(progress, text=f"Question {q_num} of {total_q}")

    if st.session_state["last_cost"]:
        st.markdown(
            f"<div class='cost-badge'>{st.session_state['last_cost']}</div>",
            unsafe_allow_html=True,
        )

    # Get last score from last assistant message
    last_score = None
    for msg in reversed(st.session_state["messages"]):
        if msg["role"] == "assistant":
            last_score = extract_score(msg["content"])
            break

    # ------------------------------------------------------------------
    # STATE A — After feedback: show action buttons
    # ------------------------------------------------------------------
    if st.session_state["awaiting_decision"]:

        is_last = st.session_state.get("is_final_answer", False)

        # Score nudge
        if last_score is not None and last_score < 6:
            st.markdown(
                f"<div class='nudge-box'>💡 You scored {last_score}/10 — "
                f"consider improving your answer!</div>",
                unsafe_allow_html=True,
            )

        show_improve = last_score is None or last_score < 10

        if show_improve:
            col1, col2, col3 = st.columns([2, 2, 2])
        else:
            col1, col2 = st.columns([2, 2])

        with col1:
            # Last question: show View Results instead of Next Question
            if is_last:
                next_q = st.button("🏁 View My Results", use_container_width=True)
            else:
                next_q = st.button("✅ Next Question", use_container_width=True)
        with col2:
            skip = st.button("⏭️ Skip Question", use_container_width=True) if not is_last else False
        if show_improve:
            with col3:
                improve = st.button("✏️ Improve My Answer", use_container_width=True,
                    help="Try answering the same question again.")
        else:
            improve = False

        if improve:
            st.session_state["awaiting_decision"] = False
            st.session_state["is_final_answer"] = False
            st.rerun()

        if next_q:
            if is_last:
                st.session_state["interview_done"] = True
                st.session_state["awaiting_decision"] = False
                st.rerun()
            else:
                st.session_state["messages"].append({
                    "role": "user",
                    "content": "I am satisfied with my answer. Please move to the next question.",
                })
                with st.spinner("Loading next question..."):
                    reply = call_openai(get_system_prompt(), st.session_state["messages"])
                # Strip feedback from next-question reply — only store the question
                reply = re.sub(r"---FEEDBACK---.*?---END FEEDBACK---", "", reply, flags=re.DOTALL).strip()
                st.session_state["messages"].append({"role": "assistant", "content": reply})
                new_count = st.session_state["question_count"] + 1
                # Hard cap — never exceed total questions
                st.session_state["question_count"] = min(new_count, total_q)
                st.session_state["attempt"] = 0
                st.session_state["awaiting_decision"] = False
                st.session_state["is_final_answer"] = False
                st.rerun()

        if skip:
            # If this is the last question, skip goes to summary directly
            if q_num >= total_q:
                st.session_state["interview_done"] = True
                st.session_state["awaiting_decision"] = False
                st.rerun()
            # Skip: fetch next question only, no feedback shown
            with st.spinner("Moving to next question..."):
                skip_msgs = st.session_state["messages"] + [{
                    "role": "user", "content": "Please skip this question and ask the next one."
                }]
                reply = call_openai(get_system_prompt(), skip_msgs)
            reply_clean = re.sub(r"---FEEDBACK---.*?---END FEEDBACK---", "", reply, flags=re.DOTALL).strip()
            st.session_state["messages"].append({"role": "user", "content": "*(skipped)*"})
            st.session_state["messages"].append({"role": "assistant", "content": reply_clean})
            st.session_state["question_count"] += 1
            st.session_state["attempt"] = 0
            st.session_state["awaiting_decision"] = False
            st.session_state["is_final_answer"] = False
            st.rerun()

    # ------------------------------------------------------------------
    # STATE B — Answer input box
    # ------------------------------------------------------------------
    else:
        if attempt > 0:
            st.markdown(
                f"<div style='font-size:0.82rem;color:#555;margin-bottom:8px;'>"
                f"Attempt {attempt + 1} — try to improve your previous answer based on the feedback above.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### ✍️ Your Answer")
        user_input = st.text_area(
            "Answer", placeholder="Write your answer here...",
            height=130, label_visibility="collapsed",
            key=f"answer_{q_num}_{attempt}",
        )

        col1, col2 = st.columns([2, 5])
        with col1:
            submit = st.button("📨 Submit Answer", use_container_width=True)
        with col2:
            skip_early = st.button("⏭️ Skip Question", use_container_width=False)

        if submit:
            if not user_input.strip():
                st.warning("⚠️ Please write an answer before submitting.")
            else:
                is_safe, reason = validate_input(user_input)
                if not is_safe:
                    st.error(reason)
                else:
                    is_last = (q_num == total_q)
                    st.session_state["is_final_answer"] = is_last
                    st.session_state["messages"].append({"role": "user", "content": user_input})
                    with st.spinner("Evaluating your answer..."):
                        reply = call_openai(
                            get_system_prompt(is_last_question=is_last),
                            st.session_state["messages"]
                        )

                    # Always strip the next question block from the reply —
                    # question_count is NEVER incremented here.
                    # The next question is fetched fresh when user clicks Next Question.
                    reply = re.sub(
                        r"---NEXT QUESTION---.*?---END QUESTION---", "",
                        reply, flags=re.DOTALL
                    ).strip()

                    st.session_state["messages"].append({"role": "assistant", "content": reply})

                    score = extract_score(reply)
                    if score is not None:
                        if attempt == 0:
                            st.session_state["scores"].append(score)
                        elif st.session_state["scores"]:
                            st.session_state["scores"][-1] = max(
                                st.session_state["scores"][-1], score
                            )

                    st.session_state["attempt"] += 1
                    st.session_state["awaiting_decision"] = True
                    st.rerun()

        if skip_early:
            # If this is the last question, skip goes to summary directly
            if q_num >= total_q:
                st.session_state["interview_done"] = True
                st.rerun()
            # Skip: fetch next question only, no feedback
            with st.spinner("Moving to next question..."):
                skip_msgs = st.session_state["messages"] + [{
                    "role": "user", "content": "Please skip this question and ask the next one."
                }]
                reply = call_openai(get_system_prompt(), skip_msgs)
            reply_clean = re.sub(r"---FEEDBACK---.*?---END FEEDBACK---", "", reply, flags=re.DOTALL).strip()
            st.session_state["messages"].append({"role": "user", "content": "*(skipped)*"})
            st.session_state["messages"].append({"role": "assistant", "content": reply_clean})
            st.session_state["question_count"] += 1
            st.session_state["attempt"] = 0
            st.session_state["awaiting_decision"] = False
            st.session_state["is_final_answer"] = False
            st.rerun()



# ======================================================================
# SECTION 3 — SUMMARY
# ======================================================================
if st.session_state["interview_done"]:

    st.markdown("---")
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif;color:#1a1a2e;'>🏁 Session Summary</h2>",
        unsafe_allow_html=True,
    )

    scores = st.session_state["scores"]
    avg_score = sum(scores) / len(scores) if scores else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Score", f"{avg_score:.1f}/10")
    with col2:
        st.metric("Best Answer", f"{max(scores) if scores else 0}/10")
    with col3:
        st.metric("Questions Completed", len(scores))

    if avg_score >= 8:
        msg, c = "🌟 Outstanding! You're well-prepared for this interview.", "#005a8e"
    elif avg_score >= 6:
        msg, c = "✅ Good work! A bit more practice and you'll be ready.", "#007a59"
    elif avg_score >= 4:
        msg, c = "📈 Decent effort — review the improved answers and try again.", "#7a5000"
    else:
        msg, c = "💪 Keep practising! Study the model answers and focus on structure.", "#a03000"

    st.markdown(
        f"<div class='summary-box' style='border-left:4px solid {c};color:{c};'>{msg}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**📊 Score Breakdown**")
    for i, s in enumerate(scores, 1):
        color = "#3a9fd4" if s >= 7 else "#56B4E9" if s >= 5 else "#D55E00"
        st.markdown(
            f"<div style='font-size:0.9rem;margin:6px 0;display:flex;align-items:center;gap:10px;'>"
            f"<span style='width:28px;color:#555;'>Q{i}</span>"
            f"<div style='flex:1;background:#e8ecf4;border-radius:4px;height:12px;overflow:hidden;'>"
            f"<div style='width:{s*10}%;background:{color};height:100%;border-radius:4px;'></div></div>"
            f"<span style='width:44px;color:#333;font-weight:500;'>{s}/10</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(format_session_cost(st.session_state["cost_log"]))
    # Gap analysis / JD summary — shown when available
    if st.session_state.get("gap_analysis"):
        st.markdown("---")
        st.markdown("**🔎 Gap Analysis**")
        st.markdown(
            "<div style='background:#ffffff;border:1px solid #d4d8e8;border-left:4px solid #3a9fd4;"
            "border-radius:10px;padding:16px 20px;color:#1a1a2e;'>"
            + html_lib.escape(st.session_state["gap_analysis"]).replace("\n", "<br>") +
            "</div>",
            unsafe_allow_html=True,
        )
    elif st.session_state.get("jd_summary"):
        st.markdown("---")
        st.markdown("**📄 Job Description Analysis**")
        st.markdown(
            "<div style='background:#ffffff;border:1px solid #d4d8e8;border-left:4px solid #3a9fd4;"
            "border-radius:10px;padding:16px 20px;color:#1a1a2e;'>"
            + html_lib.escape(st.session_state["jd_summary"]).replace("\n", "<br>") +
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**📥 Download Session Report**")

    html_report = build_html_report(
        role=role,
        interview_type=interview_type,
        difficulty=difficulty,
        technique=technique_label,
        messages=st.session_state["messages"],
        scores=st.session_state["scores"],
        jd_summary=st.session_state["jd_summary"],
        cv_summary=st.session_state["cv_summary"],
        gap_analysis=st.session_state["gap_analysis"],
    )
    filename = f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    st.download_button(
        label="⬇️ Download Session Report (.html)",
        data=html_report, file_name=filename, mime="text/html",
    )

    if st.button("🔄 Start a New Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
