"""
Hypothesis Generator - Streamlit entry point.

Two visual states:
    State 1 (no KB): Clean onboarding — title, PDF count, Build button,
                     brief description of what the app does.
    State 2 (KB ready): Full interface — tabs, hypotheses section, chat.

Run locally:
    python -m streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import config
from src.ingest import build_vectorstore, load_existing_vectorstore, load_ingestion_statuses, load_temporal_counts, PaperStatus
from src.retrievers import hybrid_search, ScoredChunk
from src.tools import run_full_pipeline, run_metadata_extraction, run_gap_analysis
from src.cost_tracking import get_tracker
from src.query_guard import validate_query
from src.exports import export_to_pdf

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
load_dotenv()

# Optional file logging — set LOG_FILE=app.log in .env to enable.
# Writes all INFO logs to the specified file AND to the terminal.
_log_file = os.getenv("LOG_FILE")
if _log_file:
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            _logging.FileHandler(_log_file),
            _logging.StreamHandler(),
        ],
    )

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colorblind-friendly palette:
#   Action/primary: #6d4cad (purple)
#   Success:        #1a8a7a (teal — distinct from blue for deuteranopia)
#   Warning:        #c47f17 (amber — distinct from purple/teal)
#   Info:           #2b6cb0 (blue)
#   No red anywhere.
st.markdown(
    """
    <style>
    [data-testid="collapsedControl"] { display: none !important; }
    .block-container { padding-top: 1.5rem; padding-bottom: 0.5rem; }
    button[kind="primary"] {
        background-color: #9d7fc7 !important;
        border-color: #9d7fc7 !important;
    }
    button[kind="primary"]:hover {
        background-color: #8b6bb7 !important;
        border-color: #8b6bb7 !important;
    }
    /* Hide the Streamlit running/stop indicators top-right */
    [data-testid="stStatusWidget"] { display: none !important; }
    .stTabs [data-baseweb="tab-panel"] { min-height: 0px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_existing_vectorstore()
if "ingestion_statuses" not in st.session_state:
    st.session_state.ingestion_statuses = []
# If KB exists from a previous session but session state is empty (app restart),
# restore ingestion statuses from the JSON saved at build time.
if st.session_state.vectorstore is not None and not st.session_state.ingestion_statuses:
    st.session_state.ingestion_statuses = load_ingestion_statuses()
if "temporal_counts" not in st.session_state:
    st.session_state.temporal_counts = load_temporal_counts()
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []
if "metadata_results" not in st.session_state:
    st.session_state.metadata_results = []
if "gap_results" not in st.session_state:
    st.session_state.gap_results = {}
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""
if "query_warning" not in st.session_state:
    st.session_state.query_warning = ""

kb_exists = st.session_state.vectorstore is not None
papers_dir = Path(config.PAPERS_DIR)
pdf_files = list(papers_dir.glob("*.pdf")) if papers_dir.exists() else []

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def fetch_chunks(vectorstore, paper_id=None, section_type=None):
    where = {}
    filters = []
    if paper_id and paper_id != "(any)":
        filters.append({"paper_id": paper_id})
    if section_type and section_type != "(any)":
        filters.append({"section_type": section_type})
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}
    try:
        return vectorstore.get(where=where if where else None), None
    except Exception as e:
        return {"documents": [], "metadatas": []}, str(e)

# =============================================================================
# SIDEBAR (static — session stats + export)
# =============================================================================
with st.sidebar:
    st.header("📊 Session Stats")
    tracker = get_tracker()
    st.metric("Input tokens", f"{tracker.total_input_tokens:,}")
    st.metric("Output tokens", f"{tracker.total_output_tokens:,}")
    st.metric("Est. cost (USD)", f"${tracker.total_cost_usd:.4f}")
    if tracker.calls:
        with st.expander(f"📜 Call log ({len(tracker.calls)} calls)"):
            for c in tracker.calls:
                st.caption(
                    f"**{c.timestamp}** · {c.model} · {c.summary}\n"
                    f"  in={c.input_tokens:,} out={c.output_tokens:,} · ${c.estimated_cost_usd:.6f}"
                )
    st.divider()
    st.subheader("📥 Export")

    # Export activates only after hypothesis is generated
    gap = st.session_state.get("gap_results", {})
    has_hypothesis = bool(gap and gap.get("hypotheses"))
    last_topic = st.session_state.get("last_topic", "query")

    if has_hypothesis:
        # Attach PubMed results from session state into hypothesis dict
        hyp = gap["hypotheses"][0]
        hyp_id = hyp.get("id", "H1")
        pubmed_key = f"pubmed_result_{hyp_id}"
        if pubmed_key in st.session_state:
            hyp["pubmed"] = st.session_state[pubmed_key]

        try:
            pdf_bytes = export_to_pdf(last_topic, gap)
            st.download_button(
                label="Export as PDF",
                data=pdf_bytes,
                file_name=f"hypothesis_{last_topic[:30].replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        except Exception as e:
            st.caption(f"Export failed: {e}")
    else:
        st.button(
            "Export as PDF",
            disabled=True,
            help="Available after hypothesis is generated",
            use_container_width=True,
        )


# =============================================================================
# STATE 1 — ONBOARDING (no knowledge base yet)
# =============================================================================
if not kb_exists:
    st.title(config.APP_TITLE)
    st.caption("For scientific research projects.")

    st.divider()

    st.caption(f"📄 {len(pdf_files)} PDF(s) detected in your library")

    if pdf_files:
        if st.button("Build knowledge base", type="primary"):
            st.caption("⏱ Please be patient — this may take a few minutes.")
            with st.spinner("Building knowledge base from your papers..."):
                try:
                    vs, statuses = build_vectorstore()
                    st.session_state.vectorstore = vs
                    st.session_state.ingestion_statuses = statuses
                    st.session_state.temporal_counts = load_temporal_counts()
                    st.rerun()
                except FileNotFoundError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.warning(f"Ingestion failed: {e}")
    else:
        st.warning("No PDFs found. Add papers to `data/papers/` and refresh.")

    st.divider()
    st.caption(f"⚠️ {config.DISCLAIMER}")


# =============================================================================
# STATE 2 — FULL INTERFACE (knowledge base ready)
# =============================================================================
else:
    total_chunks = sum(s.chunk_count for s in st.session_state.ingestion_statuses)
    paper_count = len([s for s in st.session_state.ingestion_statuses if s.status != "failed"])

    # --- Title ---
    st.title(config.APP_TITLE)
    st.caption("For scientific research projects.")
    st.success(f"✅ Knowledge base ready · 📄 {len(pdf_files)} PDFs · 📦 {total_chunks} chunks")

    # --- Query input above tabs ---
    q_col, btn_col = st.columns([4, 1])
    with q_col:
        query_input = st.text_input(
            label="query",
            label_visibility="collapsed",
            placeholder="Enter a research topic, keywords or question",
            key="query_text_input",
        )
    with btn_col:
        analyse_clicked = st.button("Analyse →", type="primary", use_container_width=True)

    # Two placeholders: top one shows persistent patience note,
    # bottom one updates per step — both sit right under the query row.
    patience_placeholder = st.empty()
    progress_placeholder = st.empty()

    # Show query validation warning if needed
    if "query_warning" in st.session_state and st.session_state.query_warning:
        st.warning(st.session_state.query_warning)

    st.divider()

    # --- Tool Output Tabs — Generated Hypothesis first ---
    tab_hypothesis, tab_sources, tab_ingestion, tab_meta, tab_gap = st.tabs(
        ["🧠 Generated Hypothesis", "📚 Sources", "🔨 Ingestion", "📋 Metadata", "🔍 Gap Analysis"]
    )

    # ---- Sources tab ----
    with tab_sources:
        if st.session_state.retrieved_chunks:
            st.caption(f"Showing {len(st.session_state.retrieved_chunks)} most relevant chunks out of {total_chunks}")
            for i, sc in enumerate(st.session_state.retrieved_chunks):
                temporal = sc.document.metadata.get("temporal_lean", "?")
                past_sim = sc.document.metadata.get("past_similarity", 0)
                future_sim = sc.document.metadata.get("future_similarity", 0)
                with st.expander(
                    f"#{i+1} · **{sc.paper_id}** · _{sc.section_type}_ · "
                    f"[{temporal}] · RRF: {sc.rrf_score:.4f}"
                ):
                    st.text(sc.text[:600] + ("..." if len(sc.text) > 600 else ""))
                    st.caption(
                        f"Semantic: {sc.semantic_score:.3f} · "
                        f"BM25: {sc.bm25_score:.2f} · "
                        f"Past sim: {past_sim:.3f} · "
                        f"Future sim: {future_sim:.3f}"
                    )
        else:
            st.caption("Retrieved chunks with section labels and similarity scores.")
            st.markdown("_Enter a query below to retrieve relevant chunks from your papers._")

    # ---- Ingestion tab ----
    with tab_ingestion:
        if st.session_state.ingestion_statuses:
            # Show temporal tagging summary if available
            tc = st.session_state.get("temporal_counts", {})
            if tc and tc.get("total", 0) > 0:
                st.caption(
                    f"📊 Temporal tagging: "
                    f"**{tc.get('past', 0)} past** · "
                    f"**{tc.get('future', 0)} future** · "
                    f"**{tc.get('neutral', 0)} neutral** "
                    f"(out of {tc.get('total', 0)} chunks)"
                )
                st.divider()

            for s in st.session_state.ingestion_statuses:
                icon = {"ingested": "✅", "partial": "⚠️", "failed": "⛔"}[s.status]
                with st.expander(f"{icon} {s.paper_id} ({s.chunk_count} chunks)"):
                    if s.sections_found:
                        for section, found in s.sections_found.items():
                            badge = "✅ explicit" if found else "⚠️ fallback"
                            st.markdown(f"- **{section}**: {badge}")
                    if s.note:
                        st.caption(s.note)

            st.divider()
            st.markdown("**Quick chunk preview:**")
            paper_ids = ["(any)"] + sorted({
                s.paper_id for s in st.session_state.ingestion_statuses if s.status != "failed"
            })
            section_types = ["(any)", "abstract", "introduction", "discussion", "limitations_future"]
            fc1, fc2 = st.columns(2)
            with fc1:
                selected_paper = st.selectbox("Paper", paper_ids, key="chunk_paper")
            with fc2:
                selected_section = st.selectbox("Section", section_types, key="chunk_section")

            result, err = fetch_chunks(st.session_state.vectorstore, selected_paper, selected_section)
            if err:
                st.info(f"Could not read chunks: {err}")
            else:
                docs = result.get("documents") or []
                metas = result.get("metadatas") or []
                st.caption(f"Total matching: **{len(docs)}** chunks")
                preview_count = min(3, len(docs))
                if docs:
                    for doc, meta in zip(docs[:preview_count], metas[:preview_count]):
                        with st.expander(
                            f"{meta.get('paper_id','?')} · {meta.get('section_type','?')} · chunk {meta.get('chunk_index','?')}"
                        ):
                            st.text(doc[:500] + ("..." if len(doc) > 500 else ""))
                    if len(docs) > preview_count:
                        st.caption(f"_{len(docs) - preview_count} more not shown._")
                else:
                    st.markdown("_No chunks match._")
        else:
            st.markdown("_Build the knowledge base to see ingestion results._")

    # ---- Metadata tab ----
    with tab_meta:
        if st.session_state.metadata_results:
            for m in st.session_state.metadata_results:
                pid = m.get("paper_id", "unknown")
                with st.expander(f"📄 {pid}"):
                    # Show only the 4 most important fields as brief bullets.
                    # The other 4 fields (research_question, methods, discussion,
                    # future_recommendations) remain in the JSON — they are used
                    # by the gap analysis pipeline but hidden from this view.
                    st.markdown(f"- **Topic:** {m.get('topic', 'N/A')}")
                    st.markdown(f"- **Hypothesis:** {m.get('hypothesis', 'N/A')}")
                    st.markdown(f"- **Key findings:** {m.get('key_findings', 'N/A')}")
                    st.markdown(f"- **Limitations:** {m.get('limitations', 'N/A')}")
        else:
            st.markdown("_Enter a query below to extract structured metadata from your papers._")

    # ---- Gap Analysis tab ----
    with tab_gap:
        gap = st.session_state.gap_results
        if gap and "past_summary" in gap and gap["past_summary"]:
            # Literature gap score with blue-toned label
            lit_gap = gap.get("literature_gap", {})
            if lit_gap:
                color = lit_gap.get("color", "#4a90d9")
                label = lit_gap.get("label", "")
                score = lit_gap.get("score", 0.0)
                sim = lit_gap.get("similarity", 0.0)
                st.markdown(
                    f"**Literature gap:** "
                    f"<span style='color:{color}; font-weight:600;'>{label}</span> "
                    f"(gap score {score:.2f}, similarity {sim:.2f})",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Higher gap score = past findings and future recommendations differ higher"
                )
                st.divider()

            # Two-column table: Past | Future (no bullet symbols, clean rows)
            past_items = gap.get("past_summary", [])
            future_items = gap.get("future_summary", [])
            
            # Build table rows — pad shorter list with empty strings
            max_rows = max(len(past_items), len(future_items))
            past_padded = past_items + [""] * (max_rows - len(past_items))
            future_padded = future_items + [""] * (max_rows - len(future_items))
            
            # Markdown table header
            table_md = "| **Past — What Has Been Done** | **Future — What Is Recommended** |\n"
            table_md += "|---|---|\n"
            
            # Table rows
            for past, future in zip(past_padded, future_padded):
                table_md += f"| {past} | {future} |\n"
            
            st.markdown(table_md)
        else:
            st.markdown("_Enter a query below to analyse gaps between past research and future directions._")

    # ---- Generated Hypothesis tab ----
    with tab_hypothesis:
        gap = st.session_state.gap_results
        if gap and "hypotheses" in gap and gap["hypotheses"]:
            for h in gap["hypotheses"]:
                statement = h.get("statement", "")
                if not statement:
                    continue

                orig_score = h.get("originality_score", None)
                grade_label = h.get("grade_label", "")
                grade_color = h.get("grade_color", config.BLUE_GRADE_COLORS["moderate"])
                max_sim = h.get("max_similarity_to_past", 0.0)
                hyp_id = h.get("id", "H1")
                pubmed_state_key = f"pubmed_result_{hyp_id}"

                with st.container(border=True):
                    # 1. Hypothesis statement
                    st.markdown(f"**{statement}**")

                    # 2. Originality score (blue tone)
                    if orig_score is not None and grade_label:
                        st.markdown(
                            f"<span style='color:{grade_color}; font-weight:600;'>"
                            f"{grade_label}</span> "
                            f"(originality {orig_score:.2f}, similarity to past {max_sim:.2f})",
                            unsafe_allow_html=True,
                        )

                    # 2b. Scientific plausibility score
                    plausibility = h.get("plausibility", {})
                    if plausibility and plausibility.get("average_score") is not None:
                        avg = plausibility["average_score"]
                        verdict = plausibility.get("verdict", "")
                        p_color = config.BLUE_GRADE_COLORS[
                            "very" if avg >= 4.0 else "less" if avg < 2.5 else "moderate"
                        ]
                        st.markdown(
                            f"<span style='color:{p_color}; font-weight:600;'>"
                            f"Scientific plausibility: {avg:.1f} / 5</span>"
                            + (f" — {verdict}" if verdict else ""),
                            unsafe_allow_html=True,
                        )
                        if avg < 2.0:
                            st.caption(
                                "Low plausibility score may indicate the query topic "
                                "is not well represented in the local knowledge base. "
                                "Consider trying a more specific or relevant research topic."
                            )

                    # 3. Supporting citations
                    papers_cited = h.get("supported_by", [])
                    if papers_cited:
                        st.markdown("**Supporting papers:**")
                        for pid in papers_cited:
                            st.markdown(f"- {pid}")

                    # 4. Suggested approach
                    approach = h.get("suggested_approach", [])
                    if isinstance(approach, str):
                        approach = [approach] if approach.strip() else []
                    if approach:
                        st.markdown("**Suggested approach:**")
                        for step in approach:
                            st.markdown(f"- {step}")

                    # 5. PubMed check — at the bottom of the card
                    st.divider()
                    btn_col, cap_col = st.columns([1, 3])
                    with btn_col:
                        btn_clicked = st.button(
                            "PubMed check",
                            key=f"pubmed_btn_{hyp_id}",
                            type="primary",
                        )
                    with cap_col:
                        st.caption("Check for PubMed papers within last 5 years and see originality of the hypothesis")

                    if btn_clicked:
                        from src.pubmed_tool import check_pubmed_freshness
                        from langchain_openai import OpenAIEmbeddings
                        with st.spinner("Checking PubMed..."):
                            embeddings_client = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
                            st.session_state[pubmed_state_key] = check_pubmed_freshness(
                                statement,
                                embeddings_client,
                                topic=st.session_state.get("last_topic", ""),
                            )

                    # PubMed results
                    pubmed = st.session_state.get(pubmed_state_key, {})
                    pm_status = pubmed.get("status", "")

                    if pm_status == "ok":
                        pm_color = pubmed.get("color", config.BLUE_GRADE_COLORS["moderate"])
                        pm_label = pubmed.get("label", "")
                        pm_sim = pubmed.get("max_similarity", 0.0)
                        pm_orig = round(1.0 - pm_sim, 2)
                        pm_total = pubmed.get("total_found", 0)
                        pm_compared = pubmed.get("papers_compared", 0)
                        pm_query = pubmed.get("query_used", "")
                        st.markdown(
                            f"**PubMed check (last 5 years):** "
                            f"<span style='color:{pm_color}; font-weight:600;'>"
                            f"{pm_label}</span> "
                            f"(originality: {pm_orig:.2f}, similarity {pm_sim:.2f})",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"Found {pm_total} papers for \"{pm_query}\" · compared top {pm_compared}"
                        )
                        matches = pubmed.get("matches", [])
                        if matches:
                            st.markdown("**Matching recent papers:**")
                            for m in matches:
                                title = m.get("title", "Untitled")
                                year = m.get("year", "")
                                url = m.get("url", "")
                                year_str = f" ({year})" if year else ""
                                if url:
                                    st.markdown(f"- [{title}]({url}){year_str}")
                                else:
                                    st.markdown(f"- {title}{year_str}")
                    elif pm_status == "no_results":
                        pm_query = pubmed.get("query_used", "")
                        st.markdown(
                            f"**PubMed check:** No recent matches found for \"{pm_query}\" — "
                            f"hypothesis may be genuinely novel."
                        )
                    elif pm_status == "error":
                        st.markdown(
                            f"**PubMed check:** ⚠️ {pubmed.get('message', 'PubMed is not responding right now.')}"
                        )
        else:
            st.markdown("_Enter a research topic above to generate a novel hypothesis._")

    # --- Analyse button handler ---
    if analyse_clicked and query_input:
        st.session_state.query_warning = ""

        is_valid, warning_msg = validate_query(query_input)
        if not is_valid:
            st.session_state.query_warning = warning_msg
            st.rerun()
        else:
            st.session_state.last_topic = query_input
            st.session_state.messages.append({"role": "user", "content": query_input})

            try:
                from langchain_openai import OpenAIEmbeddings
                from src.retrievers import get_all_chunks
                from src.generate import summarize_past, summarize_future, generate_single_hypothesis
                from src.originality import score_originality_against_summary, grade_similarity, _cosine_similarity

                embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

                # Step 1
                patience_placeholder.caption("⏱ Please be patient — this may take a few minutes.")
                progress_placeholder.info("⏳ 1 / 7 · Retrieving relevant chunks from knowledge base...")
                retrieved = hybrid_search(query_input, st.session_state.vectorstore)
                st.session_state.retrieved_chunks = retrieved

                # Step 2
                progress_placeholder.info("⏳ 2 / 7 · Extracting metadata from papers...")
                metadata = run_metadata_extraction(st.session_state.vectorstore)
                st.session_state.metadata_results = metadata

                # Step 3
                progress_placeholder.info("⏳ 3 / 7 · Summarising past research and future directions...")
                all_chunks = get_all_chunks(st.session_state.vectorstore)
                past_texts = [c.page_content for c in all_chunks if c.metadata.get("temporal_lean") == "past"]
                future_texts = [c.page_content for c in all_chunks if c.metadata.get("temporal_lean") == "future"]
                paper_ids = sorted({c.metadata.get("paper_id", "?") for c in all_chunks if c.metadata.get("paper_id")})
                tested_hyps = "\n".join(
                    f"[{m.get('paper_id', '?')}] {m.get('hypothesis', '')}"
                    for m in metadata if m.get("hypothesis") and m.get("hypothesis") != "Extraction failed"
                )
                past_bullets = summarize_past("\n\n".join(past_texts), tested_hyps)
                future_bullets = summarize_future("\n\n".join(future_texts))
                past_text = "\n".join(f"- {b}" for b in past_bullets)
                future_text = "\n".join(f"- {b}" for b in future_bullets)

                # Step 4
                progress_placeholder.info("⏳ 4 / 7 · Computing literature gap score...")
                past_vec = embeddings.embed_documents([past_text])[0]
                future_vec = embeddings.embed_documents([future_text])[0]
                gap_sim = _cosine_similarity(past_vec, future_vec)
                gap_score = round(1.0 - gap_sim, 3)
                gap_grade = grade_similarity(gap_sim, context="gap")
                literature_gap = {
                    "similarity": gap_sim, "score": gap_score,
                    "grade": gap_grade["grade"], "label": gap_grade["label"], "color": gap_grade["color"],
                }

                # Step 5
                progress_placeholder.info("⏳ 5 / 7 · Generating hypothesis...")
                gen_result = generate_single_hypothesis(
                    topic=query_input, past_summary=past_text, future_summary=future_text,
                    gap_score=gap_score, gap_label=gap_grade["label"], paper_ids=list(paper_ids),
                )
                hypothesis = gen_result.get("hypothesis", {})

                # Step 6
                progress_placeholder.info("⏳ 6 / 7 · Scoring originality...")
                if hypothesis and hypothesis.get("statement"):
                    orig_results = score_originality_against_summary([hypothesis], past_text, embeddings)
                    if orig_results:
                        orig = orig_results[0]
                        hypothesis["originality_score"] = orig.originality_score
                        hypothesis["max_similarity_to_past"] = orig.max_similarity
                        hypothesis["is_original"] = orig.is_original
                        hypothesis["grade"] = orig.grade
                        hypothesis["grade_label"] = orig.grade_label
                        hypothesis["grade_color"] = orig.grade_color

                progress_placeholder.info("⏳ 7 / 7 · Judging scientific plausibility...")
                if hypothesis and hypothesis.get("statement"):
                    from src.generate import judge_scientific_plausibility
                    plausibility = judge_scientific_plausibility(
                        statement=hypothesis["statement"],
                        supported_by=hypothesis.get("supported_by", []),
                        topic=query_input,
                    )
                    hypothesis["plausibility"] = plausibility

                st.session_state.gap_results = {
                    "past_summary": past_bullets,
                    "future_summary": future_bullets,
                    "literature_gap": literature_gap,
                    "hypotheses": [hypothesis] if hypothesis else [],
                }

                progress_placeholder.empty()
                patience_placeholder.empty()

            except Exception as e:
                progress_placeholder.empty()
                patience_placeholder.empty()
                st.session_state.query_warning = f"Pipeline failed: {e}"

            st.rerun()

    # --- Disclaimer ---
    st.divider()
    st.caption(f"⚠️ {config.DISCLAIMER}")
