"""
Tool 1: extract_paper_metadata
Tool 2: analyse_gaps

These are LLM + RAG tools — they read from ChromaDB via hybrid search
and send the retrieved chunks to GPT-4o-mini for structured output.

Public API:
    - run_metadata_extraction(vectorstore) -> list[dict]
    - run_gap_analysis(topic, vectorstore, metadata_list) -> dict
    - run_full_pipeline(topic, vectorstore) -> dict
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.retrievers import (
    get_all_chunks,
    hybrid_search,
    retrieve_limitations,
    ScoredChunk,
)
from src.generate import (
    extract_paper_metadata,
    analyse_gaps_and_generate_hypotheses,
    summarize_past,
    summarize_future,
    generate_single_hypothesis,
)
from src.originality import (
    filter_genuine_gaps,
    score_originality,
    score_originality_against_summary,
    grade_similarity,
    _cosine_similarity,
)
import config

logger = logging.getLogger(__name__)


# =============================================================================
# Tool 1: Metadata extraction
# =============================================================================

def run_metadata_extraction(vectorstore: Chroma) -> list[dict[str, Any]]:
    """
    Extract structured metadata from every paper in the knowledge base.

    For each unique paper_id found in ChromaDB, all its chunks are
    retrieved, grouped, and sent to GPT-4o-mini in a single LLM call.
    This produces one metadata JSON per paper.

    Args:
        vectorstore: the ChromaDB vector store.

    Returns:
        List of metadata dictionaries (one per paper).
    """
    all_chunks = get_all_chunks(vectorstore)
    if not all_chunks:
        logger.warning("No chunks found in vector store — metadata extraction skipped")
        return []

    # Group chunks by paper_id
    papers: dict[str, list] = {}
    for chunk in all_chunks:
        pid = chunk.metadata.get("paper_id", "unknown")
        if pid not in papers:
            papers[pid] = []
        papers[pid].append(chunk)

    logger.info(f"Extracting metadata from {len(papers)} papers")

    metadata_list = []
    for paper_id, chunks in papers.items():
        # Format chunks with section labels so the LLM knows which
        # section each chunk came from
        chunks_text = ""
        for chunk in chunks:
            section = chunk.metadata.get("section_type", "unknown")
            chunks_text += f"[{section.upper()}] {chunk.page_content}\n\n"

        metadata = extract_paper_metadata(paper_id, chunks_text)
        metadata_list.append(metadata)

    return metadata_list


# =============================================================================
# Tool 2: Gap analysis + hypothesis generation
# =============================================================================

def run_gap_analysis(
    topic: str,
    vectorstore: Chroma,
    metadata_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyse gaps across all papers and generate ONE novel hypothesis.

    Summary-based pipeline (Option A):
        1. Collect past-tagged chunks + tested hypotheses from metadata
        2. Collect future-tagged chunks
        3. Summarise past into 5 bullets via LLM
        4. Summarise future into 5 bullets via LLM
        5. Compute literature gap score = 1 - cos(past_summary, future_summary)
        6. Grade the gap score (very/moderate/less)
        7. Generate ONE hypothesis from past + future summaries
        8. Score originality = 1 - cos(hypothesis, past_summary), with grade

    Args:
        topic: the user's neuroscience topic.
        vectorstore: the ChromaDB vector store.
        metadata_list: output from run_metadata_extraction().

    Returns:
        Dictionary with:
            'past_summary': list[str] (5 bullets)
            'future_summary': list[str] (5 bullets)
            'literature_gap': {similarity, score, grade, label, color}
            'hypotheses': list[dict] (exactly 1 hypothesis with originality)
    """
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    # --- Step 1: Collect past & future chunks and paper IDs ---
    all_chunks = get_all_chunks(vectorstore)

    past_chunk_texts = [
        c.page_content for c in all_chunks
        if c.metadata.get("temporal_lean") == "past"
    ]
    future_chunk_texts = [
        c.page_content for c in all_chunks
        if c.metadata.get("temporal_lean") == "future"
    ]

    logger.info(
        f"Temporal chunks: {len(past_chunk_texts)} past, "
        f"{len(future_chunk_texts)} future"
    )

    # Paper IDs for citation in the hypothesis
    paper_ids = sorted({
        c.metadata.get("paper_id", "?") for c in all_chunks
        if c.metadata.get("paper_id")
    })

    # Tested hypotheses from each paper (feeds into past summary)
    tested_hyps_lines = []
    key_findings_lines = []
    future_rec_lines = []
    limitations_lines = []

    for m in metadata_list:
        pid = m.get("paper_id", "unknown")

        h = m.get("hypothesis", "").strip()
        if h and h != "Extraction failed":
            tested_hyps_lines.append(f"[{pid}] {h}")

        kf = m.get("key_findings", "").strip()
        if kf and kf not in ("Not available in provided sections", "Extraction failed"):
            key_findings_lines.append(f"[{pid}] {kf}")

        fr = m.get("future_recommendations", "").strip()
        if fr and fr not in ("Not available in provided sections", "Extraction failed"):
            future_rec_lines.append(f"[{pid}] {fr}")

        lim = m.get("limitations", "").strip()
        if lim and lim not in ("Not available in provided sections", "Extraction failed"):
            limitations_lines.append(f"[{pid}] {lim}")

    tested_hypotheses_text = "\n".join(tested_hyps_lines)
    key_findings_text = "\n".join(key_findings_lines)
    future_recs_text = "\n".join(future_rec_lines)
    limitations_text = "\n".join(limitations_lines)

    # --- Step 2: Summarise past and future into 3 bullets each ---
    past_chunks_joined = "\n\n".join(past_chunk_texts)
    future_chunks_joined = "\n\n".join(future_chunk_texts)

    # Past summary: chunk text + tested hypotheses + key findings
    past_input = past_chunks_joined
    if key_findings_text:
        past_input += "\n\nKEY FINDINGS FROM PAPERS:\n" + key_findings_text
    past_summary_bullets = summarize_past(past_input, tested_hypotheses_text)

    # Future summary: chunk text + future recommendations + limitations
    future_input = future_chunks_joined
    if future_recs_text:
        future_input += "\n\nFUTURE RECOMMENDATIONS FROM PAPERS:\n" + future_recs_text
    if limitations_text:
        future_input += "\n\nLIMITATIONS FROM PAPERS (indicating what remains to be done):\n" + limitations_text
    future_summary_bullets = summarize_future(future_input)

    past_summary_text = "\n".join(f"- {b}" for b in past_summary_bullets)
    future_summary_text = "\n".join(f"- {b}" for b in future_summary_bullets)

    # --- Step 3: Literature gap score (past_summary vs future_summary) ---
    past_vec = embeddings.embed_documents([past_summary_text])[0]
    future_vec = embeddings.embed_documents([future_summary_text])[0]
    gap_similarity = _cosine_similarity(past_vec, future_vec)
    gap_score = 1.0 - gap_similarity
    gap_grade = grade_similarity(gap_similarity, context="gap")

    literature_gap = {
        "similarity": gap_similarity,
        "score": gap_score,
        "grade": gap_grade["grade"],
        "label": gap_grade["label"],
        "color": gap_grade["color"],
    }

    logger.info(
        f"Literature gap: similarity={gap_similarity:.3f}, "
        f"score={gap_score:.3f}, grade={gap_grade['label']}"
    )

    # --- Step 4: Generate ONE hypothesis from the summaries ---
    gen_result = generate_single_hypothesis(
        topic=topic,
        past_summary=past_summary_text,
        future_summary=future_summary_text,
        gap_score=gap_score,
        gap_label=gap_grade["label"],
        paper_ids=paper_ids,
    )
    hypothesis = gen_result.get("hypothesis", {})

    # --- Step 5: Score originality (hypothesis vs past_summary) ---
    if hypothesis and hypothesis.get("statement"):
        originality_results = score_originality_against_summary(
            [hypothesis],
            past_summary_text,
            embeddings,
        )
        if originality_results:
            orig = originality_results[0]
            hypothesis["originality_score"] = orig.originality_score
            hypothesis["max_similarity_to_past"] = orig.max_similarity
            hypothesis["is_original"] = orig.is_original
            hypothesis["grade"] = orig.grade
            hypothesis["grade_label"] = orig.grade_label
            hypothesis["grade_color"] = orig.grade_color

    # --- Step 6: Scientific plausibility judge ---
    # Scores the hypothesis on 6 dimensions (novelty, testability,
    # mechanistic coherence, citation traceability, conflict awareness,
    # usefulness). Returns an average score 0-5 and a one-sentence verdict.
    if hypothesis and hypothesis.get("statement"):
        from src.generate import judge_scientific_plausibility
        plausibility = judge_scientific_plausibility(
            statement=hypothesis["statement"],
            supported_by=hypothesis.get("supported_by", []),
            topic=topic,
        )
        hypothesis["plausibility"] = plausibility

    # PubMed freshness check is opt-in via the "PubMed check" button
    # in the UI — see app.py. Not called automatically here.

    return {
        "past_summary": past_summary_bullets,
        "future_summary": future_summary_bullets,
        "literature_gap": literature_gap,
        "hypotheses": [hypothesis] if hypothesis else [],
    }


# =============================================================================
# Full pipeline (runs both tools in sequence)
# =============================================================================

def run_full_pipeline(
    topic: str,
    vectorstore: Chroma,
) -> dict[str, Any]:
    """
    Run the complete analysis pipeline for a given topic:
        1. Retrieve relevant chunks via hybrid search
        2. Extract structured metadata per paper (Tool 1)
        3. Analyse gaps and generate hypotheses (Tool 2)

    All API calls are logged to the SessionCostTracker automatically
    (handled inside generate.py).

    Args:
        topic: the user's neuroscience topic.
        vectorstore: the ChromaDB vector store.

    Returns:
        Dictionary with keys:
            'retrieved_chunks': list[ScoredChunk] — for the Sources tab
            'metadata': list[dict] — for the Metadata tab
            'gap_analysis': dict — for Gap Analysis tab + Hypotheses section
    """
    logger.info(f"Starting full pipeline for topic: '{topic[:80]}'")

    # Step 1: Retrieve relevant chunks (for display in Sources tab)
    retrieved = hybrid_search(topic, vectorstore)

    # Step 2: Extract metadata from all papers (Tool 1)
    metadata_list = run_metadata_extraction(vectorstore)

    # Step 3: Analyse gaps and generate hypotheses (Tool 2)
    gap_result = run_gap_analysis(topic, vectorstore, metadata_list)

    logger.info("Full pipeline complete")

    return {
        "retrieved_chunks": retrieved,
        "metadata": metadata_list,
        "gap_analysis": gap_result,
    }
