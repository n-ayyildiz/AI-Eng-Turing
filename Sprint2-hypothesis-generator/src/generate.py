"""
LangChain chains for query translation, metadata extraction, and
gap analysis / hypothesis generation.

Light LangChain is used — manual chain composition, no autonomous agent.
Every step is visible, testable, and explainable.

Every LLM call is logged to the SessionCostTracker so token usage
and estimated costs are visible in the sidebar in real time.

Chains:
    - query_translation_chain: user topic → 2-3 retrieval-friendly variants
    - extraction_chain: retrieved chunks → structured JSON metadata per paper
    - gap_analysis_chain: limitation + discussion chunks → gap table + hypotheses
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from src.cost_tracking import get_tracker, count_tokens

logger = logging.getLogger(__name__)

# =============================================================================
# LLM instance (shared across all chains)
# =============================================================================

def _get_llm(temperature: float | None = None) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance configured with the main LLM model.
    A fresh instance is created each call to avoid stale connection issues.
    """
    return ChatOpenAI(
        model=config.MAIN_LLM_MODEL,
        temperature=temperature if temperature is not None else config.MAIN_LLM_TEMPERATURE,
    )


def _log_llm_call(model: str, call_type: str, summary: str, input_text: str, output_text: str) -> None:
    """
    Count tokens and log a completed LLM call to the session cost tracker.
    Called after every successful LLM invocation.
    """
    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model)
    tracker = get_tracker()
    tracker.log_call(
        model=model,
        call_type=call_type,
        summary=summary,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# =============================================================================
# Query translation
# =============================================================================

QUERY_TRANSLATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a neuroscience research assistant. Your task is to rewrite "
     "a user's research topic into {num_variants} different retrieval-friendly "
     "search queries. Each variant should approach the topic from a different "
     "angle to maximise recall when searching a scientific paper database.\n\n"
     "Rules:\n"
     "- Each query should be 5-15 words\n"
     "- Use specific neuroscience terminology\n"
     "- One variant should focus on mechanisms/methods\n"
     "- One variant should focus on limitations or gaps in the literature\n"
     "- One variant should focus on future directions or recommendations\n"
     "- Return ONLY a JSON array of strings, no other text"),
    ("human", "Topic: {topic}"),
])


def generate_query_variants(topic: str) -> list[str]:
    """
    Rewrite a user topic into multiple retrieval-friendly query variants.

    Args:
        topic: the user's neuroscience research topic.

    Returns:
        List of 2-3 query strings. Falls back to [topic] if the LLM
        call fails or returns unparseable output.
    """
    llm = _get_llm(temperature=0.3)
    chain = QUERY_TRANSLATION_PROMPT | llm | StrOutputParser()

    try:
        input_text = f"Topic: {topic}"
        raw = chain.invoke({
            "topic": topic,
            "num_variants": config.NUM_QUERY_VARIANTS,
        })
        _log_llm_call(config.MAIN_LLM_MODEL, "llm", f"query translation: {topic[:40]}", input_text, raw)

        variants = json.loads(raw.strip())
        if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
            logger.info(f"Generated {len(variants)} query variants for '{topic[:50]}'")
            return variants
    except Exception as e:
        logger.warning(f"Query translation failed, falling back to original topic: {e}")

    return [topic]


# =============================================================================
# Metadata extraction
# =============================================================================

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a neuroscience research metadata extractor. Given chunks of text "
     "from a scientific paper, extract structured metadata.\n\n"
     "The chunks come from different sections of the paper (abstract, discussion, "
     "limitations/future work). Use ALL provided chunks to build a complete picture.\n\n"
     "Return ONLY a JSON object with these fields (use 1-3 sentences per field):\n"
     '{{\n'
     '  "topic": "main research topic of the paper",\n'
     '  "hypothesis": "the specific research hypothesis tested or implied by '
     'the study — state it as a testable prediction. Look for explicit signals '
     'such as: we hypothesised, it was tested whether, we aimed to examine, '
     'the research goal was, we investigated whether, it was predicted that, '
     'the objective was to determine. If no explicit hypothesis is stated, '
     'infer the implied hypothesis from the research question, study design, '
     'and stated objectives",\n'
     '  "research_question": "the main research question addressed",\n'
     '  "methods": "key methods, study design, and analytical approach",\n'
     '  "key_findings": "most important results and statistical outcomes",\n'
     '  "discussion": "main discussion points and contribution to the literature",\n'
     '  "limitations": "study limitations and methodological constraints. '
     'If no standalone limitations section exists, extract limitations from '
     'the discussion section",\n'
     '  "future_recommendations": "suggested directions for future research"\n'
     '}}\n\n'
     "Rules:\n"
     "- If a field cannot be determined from the provided chunks, write "
     '"Not available in provided sections"\n'
     "- Be specific — include brain regions, biomarkers, population details, "
     "and statistical measures where present\n"
     "- For the hypothesis field, search across abstract, introduction cues, "
     "and discussion sections for explicit or implied hypotheses\n"
     "- Do not invent information not present in the chunks\n"
     "- Return ONLY valid JSON, no markdown fences, no preamble"),
    ("human",
     "Paper ID: {paper_id}\n\n"
     "Chunks from this paper:\n\n{chunks_text}"),
])


def extract_paper_metadata(paper_id: str, chunks_text: str) -> dict[str, Any]:
    """
    Extract structured metadata from a single paper's chunks.

    One LLM call per paper. All relevant chunks (abstract + discussion +
    limitations_future) are concatenated and sent together so the model
    has full context.

    Args:
        paper_id: filename stem identifying the paper.
        chunks_text: concatenated text of all chunks from this paper,
                     with section labels prepended to each chunk.

    Returns:
        Dictionary with metadata fields. Returns a minimal error dict
        if the LLM call fails or output is unparseable.
    """
    llm = _get_llm()
    chain = EXTRACTION_PROMPT | llm | StrOutputParser()

    try:
        input_text = f"Paper ID: {paper_id}\n\nChunks:\n{chunks_text[:500]}..."
        raw = chain.invoke({
            "paper_id": paper_id,
            "chunks_text": chunks_text,
        })
        _log_llm_call(config.MAIN_LLM_MODEL, "llm", f"metadata extraction: {paper_id}", input_text, raw)

        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        metadata = json.loads(cleaned)
        metadata["paper_id"] = paper_id
        logger.info(f"Extracted metadata for {paper_id}")
        return metadata
    except Exception as e:
        logger.error(f"Metadata extraction failed for {paper_id}: {e}")
        return {
            "paper_id": paper_id,
            "topic": "Extraction failed",
            "hypothesis": str(e),
            "research_question": "",
            "methods": "",
            "key_findings": "",
            "discussion": "",
            "limitations": "",
            "future_recommendations": "",
        }


# =============================================================================
# Gap analysis + hypothesis generation (two-stage novelty approach)
# =============================================================================

GAP_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a neuroscience research analyst specialising in identifying "
     "gaps in the literature and generating genuinely novel, testable hypotheses.\n\n"
     "IMPORTANT: The user's topic is: {topic}\n"
     "All generated hypotheses MUST be directly relevant to this specific topic. "
     "Do not drift to tangential topics that happen to appear in the papers.\n\n"
     "You are given:\n"
     "1. METADATA from multiple papers (topic, hypothesis, findings, limitations, "
     "future recommendations)\n"
     "2. VALIDATED GAP PAIRS — past-future chunk pairs that have been verified as "
     "genuinely different (low cosine similarity). These represent real research gaps.\n"
     "3. ADDITIONAL LIMITATION AND DISCUSSION CHUNKS for broader context\n\n"
     "Your task has TWO STAGES:\n\n"
     "═══ STAGE 1: IDENTIFY WHAT HAS ALREADY BEEN TESTED ═══\n"
     "List every specific hypothesis that the provided papers already tested "
     "or investigated. These come from the 'hypothesis' field in each paper's "
     "metadata. This list becomes the DO-NOT-REPEAT constraint for Stage 2.\n\n"
     "═══ STAGE 2: GENERATE GENUINELY NOVEL HYPOTHESES ═══\n"
     "Generate 3-5 novel, testable research hypotheses about {topic} that:\n"
     "- Are NOT on the Stage 1 list (must differ from all existing hypotheses)\n"
     "- Are inspired by the VALIDATED GAP PAIRS — use the genuine gaps between "
     "past findings and future recommendations as the foundation\n"
     "- Combine discussion points or limitations from at least 2 different "
     "papers in a way that no single paper already explored\n"
     "- Propose a specific NEW variable, population, method, or mechanism "
     "not tested in any of the provided papers\n"
     "- Are specific enough to design an experiment around\n"
     "- Include neuroscience domain context (brain regions, biomarkers, "
     "populations, methods)\n"
     "- Stay focused on the user's stated topic: {topic}\n\n"
     "NOVELTY TEST — for each hypothesis, ask:\n"
     "  1. Is this just restating what a paper already found? → REJECT\n"
     "  2. Is this just rephrasing an existing paper's hypothesis? → REJECT\n"
     "  3. Does this combine insights from 2+ papers in a new way? → KEEP\n"
     "  4. Does this propose something no paper tested? → KEEP\n\n"
     "Also produce a GAP TABLE:\n"
     "- PAST: Key themes from discussion sections studied across the papers "
     "(group by theme, not by paper)\n"
     "- FUTURE: Recurring gaps, limitations, and recommendations from "
     "discussion and limitations sections across the papers\n"
     "- NOVEL: For each gap, a brief suggestion of how it could be addressed\n\n"
     "Return ONLY a JSON object with this structure:\n"
     '{{\n'
     '  "existing_hypotheses": [\n'
     '    "Paper X tested: [hypothesis]",\n'
     '    "Paper Y tested: [hypothesis]"\n'
     '  ],\n'
     '  "gap_table": {{\n'
     '    "past": ["theme 1: description", "theme 2: description"],\n'
     '    "future": ["gap 1: description", "gap 2: description"],\n'
     '    "novel": ["suggestion 1", "suggestion 2"]\n'
     '  }},\n'
     '  "hypotheses": [\n'
     '    {{\n'
     '      "id": "H1",\n'
     '      "statement": "...",\n'
     '      "research_gap": "what specific gap or limitation this addresses",\n'
     '      "novelty_rationale": "why this is not a restatement of existing work — '
     'what new variable, population, method, or mechanism is proposed",\n'
     '      "supported_by": ["paper_id_1", "paper_id_2"],\n'
     '      "suggested_approach": "brief description of how to test this"\n'
     '    }}\n'
     '  ]\n'
     '}}\n\n'
     "Rules:\n"
     "- Generate between 3 and 5 hypotheses\n"
     "- Each hypothesis must cite at least 2 papers from the provided metadata\n"
     "- Each hypothesis must pass the novelty test above\n"
     "- Do not invent papers or findings not present in the input\n"
     "- Stay focused on {topic} — do not generate hypotheses about unrelated topics\n"
     "- Be specific with neuroscience terminology\n"
     "- Return ONLY valid JSON, no markdown fences, no preamble"),
    ("human",
     "TOPIC: {topic}\n\n"
     "PAPER METADATA:\n{metadata_text}\n\n"
     "LIMITATION AND DISCUSSION CHUNKS:\n{limitations_text}"),
])


def analyse_gaps_and_generate_hypotheses(
    topic: str,
    metadata_list: list[dict[str, Any]],
    limitation_chunks_text: str,
) -> dict[str, Any]:
    """
    Analyse gaps across all papers and generate novel hypotheses.

    Two-stage approach:
        Stage 1: Identify existing hypotheses already tested (DO-NOT-REPEAT list)
        Stage 2: Generate novel hypotheses that pass the novelty test

    Args:
        topic: the user's neuroscience topic (used to anchor hypotheses).
        metadata_list: list of metadata dicts (one per paper).
        limitation_chunks_text: concatenated limitation/discussion chunks.

    Returns:
        Dictionary with 'existing_hypotheses', 'gap_table', and 'hypotheses' keys.
    """
    metadata_text = ""
    for m in metadata_list:
        metadata_text += (
            f"--- {m.get('paper_id', 'unknown')} ---\n"
            f"Topic: {m.get('topic', 'N/A')}\n"
            f"Hypothesis tested: {m.get('hypothesis', 'N/A')}\n"
            f"Research question: {m.get('research_question', 'N/A')}\n"
            f"Methods: {m.get('methods', 'N/A')}\n"
            f"Key findings: {m.get('key_findings', 'N/A')}\n"
            f"Discussion: {m.get('discussion', 'N/A')}\n"
            f"Limitations: {m.get('limitations', 'N/A')}\n"
            f"Future recommendations: {m.get('future_recommendations', 'N/A')}\n\n"
        )

    llm = _get_llm(temperature=0.4)
    chain = GAP_ANALYSIS_PROMPT | llm | StrOutputParser()

    try:
        input_text = f"TOPIC: {topic}\n\nMETADATA:\n{metadata_text[:500]}...\n\nLIMITATIONS:\n{limitation_chunks_text[:500]}..."
        raw = chain.invoke({
            "topic": topic,
            "metadata_text": metadata_text,
            "limitations_text": limitation_chunks_text,
        })
        _log_llm_call(config.MAIN_LLM_MODEL, "llm", f"gap analysis: {topic[:40]}", input_text, raw)

        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(cleaned)
        logger.info(
            f"Gap analysis complete: "
            f"{len(result.get('existing_hypotheses', []))} existing, "
            f"{len(result.get('hypotheses', []))} novel"
        )
        return result
    except Exception as e:
        logger.error(f"Gap analysis failed: {e}")
        return {
            "existing_hypotheses": [],
            "gap_table": {
                "past": ["Analysis failed — see error"],
                "future": [str(e)],
                "novel": [],
            },
            "hypotheses": [],
        }


# =============================================================================
# Summarization (past and future chunks → bullet-point summaries)
# =============================================================================

PAST_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are summarising past research across multiple neuroscience papers. "
     "Given chunks of past-tagged content and the tested hypotheses from the "
     "papers in the library, produce exactly 3 concise bullet points capturing "
     "the main findings, methods used, and tested hypotheses across ALL papers.\n\n"
     "Rules:\n"
     "- Exactly 3 bullets, each ~15-25 words\n"
     "- Combine insights across papers (do not list one bullet per paper)\n"
     "- Use specific neuroscience terminology (biomarkers, methods, populations)\n"
     "- Focus on what HAS BEEN studied/found\n"
     "- Return ONLY a JSON array of 3 strings, no other text"),
    ("human",
     "PAST-TAGGED CHUNKS:\n{past_chunks_text}\n\n"
     "TESTED HYPOTHESES FROM EACH PAPER:\n{tested_hypotheses}"),
])


FUTURE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are summarising future research directions across multiple neuroscience "
     "papers. Given chunks of future-tagged content (limitations, recommendations, "
     "open questions), produce exactly 3 concise bullet points capturing the "
     "main gaps, methodological improvements needed, and suggested directions "
     "across ALL papers.\n\n"
     "Rules:\n"
     "- Exactly 3 bullets, each ~15-25 words\n"
     "- Combine recommendations across papers (do not list one bullet per paper)\n"
     "- Use specific neuroscience terminology\n"
     "- Focus on what SHOULD BE studied next / what REMAINS unknown\n"
     "- Return ONLY a JSON array of 3 strings, no other text"),
    ("human", "FUTURE-TAGGED CHUNKS:\n{future_chunks_text}"),
])


def _parse_bullets(raw: str) -> list[str]:
    """Parse a JSON array of strings from LLM output."""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        bullets = json.loads(cleaned)
        if isinstance(bullets, list) and all(isinstance(b, str) for b in bullets):
            return bullets
    except Exception as e:
        logger.warning(f"Bullet parse failed, returning raw text: {e}")
    return [cleaned]


def summarize_past(past_chunks_text: str, tested_hypotheses: str) -> list[str]:
    """
    Summarise all past-tagged content + tested hypotheses into 5 bullets.

    One LLM call. Cost ~$0.003 depending on input size.
    """
    if not past_chunks_text.strip() and not tested_hypotheses.strip():
        return ["No past content available."]

    llm = _get_llm(temperature=0.2)
    chain = PAST_SUMMARY_PROMPT | llm | StrOutputParser()

    try:
        input_text = f"PAST:\n{past_chunks_text[:500]}...\n\nHYPS:\n{tested_hypotheses[:500]}..."
        raw = chain.invoke({
            "past_chunks_text": past_chunks_text,
            "tested_hypotheses": tested_hypotheses,
        })
        _log_llm_call(config.MAIN_LLM_MODEL, "llm", "summarize past", input_text, raw)
        bullets = _parse_bullets(raw)
        logger.info(f"Past summary: {len(bullets)} bullets")
        return bullets
    except Exception as e:
        logger.error(f"Past summarization failed: {e}")
        return [f"Past summarization failed: {e}"]


def summarize_future(future_chunks_text: str) -> list[str]:
    """
    Summarise all future-tagged content into 5 bullets.

    One LLM call. Cost ~$0.003 depending on input size.
    """
    if not future_chunks_text.strip():
        return ["No future-tagged content available."]

    llm = _get_llm(temperature=0.2)
    chain = FUTURE_SUMMARY_PROMPT | llm | StrOutputParser()

    try:
        input_text = f"FUTURE:\n{future_chunks_text[:500]}..."
        raw = chain.invoke({"future_chunks_text": future_chunks_text})
        _log_llm_call(config.MAIN_LLM_MODEL, "llm", "summarize future", input_text, raw)
        bullets = _parse_bullets(raw)
        logger.info(f"Future summary: {len(bullets)} bullets")
        return bullets
    except Exception as e:
        logger.error(f"Future summarization failed: {e}")
        return [f"Future summarization failed: {e}"]


# =============================================================================
# LLM classification of neutral-tagged chunks (Option B fallback)
# =============================================================================

NEUTRAL_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You classify a chunk of neuroscience paper text as representing "
     "PAST research findings or FUTURE research recommendations.\n\n"
     "PAST = described findings, established results, observations, "
     "what has been studied, methods used, completed investigations.\n"
     "FUTURE = limitations of the present work, recommendations for "
     "future research, unresolved questions, suggested investigations.\n\n"
     "If the chunk is genuinely a mix of both with no clear lean, return 'neutral'.\n"
     "Otherwise return exactly one word: 'past' or 'future'.\n"
     "Return ONLY the word, no punctuation, no explanation."),
    ("human", "Chunk:\n{chunk_text}"),
])


def classify_neutral_chunk(chunk_text: str) -> str:
    """
    Ask the LLM to classify a chunk as past/future/neutral.

    Used as a fallback when the embedding-based tagging returned 'neutral'
    (past_similarity and future_similarity within TEMPORAL_NEUTRAL_MARGIN).

    Returns:
        One of "past", "future", or "neutral".
    """
    llm = _get_llm(temperature=0.0)
    chain = NEUTRAL_CLASSIFY_PROMPT | llm | StrOutputParser()

    try:
        input_text = f"Chunk:\n{chunk_text[:300]}..."
        raw = chain.invoke({"chunk_text": chunk_text})
        _log_llm_call(config.MAIN_LLM_MODEL, "llm", "classify neutral chunk", input_text, raw)

        answer = raw.strip().lower().strip(".,;:'\"")
        if answer in ("past", "future", "neutral"):
            return answer
        logger.warning(f"Unexpected classification response: {raw!r} — defaulting to neutral")
        return "neutral"
    except Exception as e:
        logger.error(f"Neutral chunk classification failed: {e}")
        return "neutral"


# =============================================================================
# Single-hypothesis generation from past + future summaries
# =============================================================================

SINGLE_HYPOTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a neuroscience research analyst. Given a PAST SUMMARY (what "
     "has been studied across the literature) and a FUTURE SUMMARY (what "
     "is recommended for future research), generate ONE novel, testable "
     "research hypothesis about the user's topic: {topic}.\n\n"
     "Requirements for the hypothesis:\n"
     "- Stated as 1-2 concise sentences (a testable prediction)\n"
     "- Must not restate any past finding — must be genuinely novel\n"
     "- Must be inspired by the gap between past and future summaries\n"
     "- Must propose a specific new variable, population, method, or mechanism\n"
     "- Must stay focused on the user's topic: {topic}\n"
     "- Must include concrete neuroscience terminology (brain regions, "
     "biomarkers, populations, methods)\n\n"
     "Also provide:\n"
     "- supported_by: list of paper IDs from the library that relate to the hypothesis\n"
     "- suggested_approach: 1-3 short bullet points describing how to test it\n\n"
     "Return ONLY a JSON object with this structure:\n"
     '{{\n'
     '  "hypothesis": {{\n'
     '    "id": "H1",\n'
     '    "statement": "...",\n'
     '    "supported_by": ["paper_id_1", "paper_id_2"],\n'
     '    "suggested_approach": ["approach bullet 1", "approach bullet 2"]\n'
     '  }}\n'
     '}}\n'
     "No markdown fences, no preamble."),
    ("human",
     "TOPIC: {topic}\n\n"
     "PAST SUMMARY:\n{past_summary}\n\n"
     "FUTURE SUMMARY:\n{future_summary}\n\n"
     "LITERATURE GAP SCORE: {gap_score:.3f} ({gap_label})\n"
     "(Higher = bigger gap between past and future, more room for novel work)\n\n"
     "AVAILABLE PAPER IDs: {paper_ids}"),
])


def generate_single_hypothesis(
    topic: str,
    past_summary: str,
    future_summary: str,
    gap_score: float,
    gap_label: str,
    paper_ids: list[str],
) -> dict[str, Any]:
    """
    Generate ONE novel hypothesis from past/future summaries.

    Single LLM call. Returns a dict with 'hypothesis' containing:
        id, statement, supported_by, suggested_approach.
    """
    llm = _get_llm(temperature=0.4)
    chain = SINGLE_HYPOTHESIS_PROMPT | llm | StrOutputParser()

    try:
        input_text = f"TOPIC: {topic}\nPAST: {past_summary[:300]}...\nFUTURE: {future_summary[:300]}..."
        raw = chain.invoke({
            "topic": topic,
            "past_summary": past_summary,
            "future_summary": future_summary,
            "gap_score": gap_score,
            "gap_label": gap_label,
            "paper_ids": ", ".join(paper_ids),
        })
        _log_llm_call(config.MAIN_LLM_MODEL, "llm", f"single hypothesis: {topic[:40]}", input_text, raw)

        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(cleaned)
        logger.info("Single hypothesis generated")
        return result
    except Exception as e:
        logger.error(f"Single hypothesis generation failed: {e}")
        return {
            "hypothesis": {
                "id": "H1",
                "statement": f"Hypothesis generation failed: {e}",
                "supported_by": [],
                "suggested_approach": [],
            }
        }


# =============================================================================
# Scientific plausibility judge (LLM-as-judge, 6 dimensions)
# =============================================================================

PLAUSIBILITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "A scientific hypothesis is provided below, along with the paper IDs "
     "that were cited as supporting evidence.\n\n"
     "Score the hypothesis on each of the following 6 dimensions. "
     "Each dimension is scored 1 (poor) to 5 (excellent).\n\n"
     "Dimensions:\n"
     "1. novelty — Does the hypothesis go beyond what the cited papers directly state? "
     "Does it propose something not yet tested?\n"
     "2. testability — Is the hypothesis specific enough to design an experiment around? "
     "Does it name measurable variables?\n"
     "3. mechanistic_coherence — Is the proposed biological or physiological mechanism "
     "plausible and internally consistent?\n"
     "4. citation_traceability — Can the claims in the hypothesis be traced back to "
     "the cited supporting papers? Are the citations relevant?\n"
     "5. conflict_awareness — Does the hypothesis acknowledge uncertainty, heterogeneity, "
     "or contradictions in the evidence where they exist?\n"
     "6. usefulness — Would this hypothesis be useful to a researcher designing a "
     "future study? Does it open a meaningful research direction?\n\n"
     "Return ONLY a JSON object — no markdown, no preamble:\n"
     '{{\n'
     '  "novelty": <1-5>,\n'
     '  "testability": <1-5>,\n'
     '  "mechanistic_coherence": <1-5>,\n'
     '  "citation_traceability": <1-5>,\n'
     '  "conflict_awareness": <1-5>,\n'
     '  "usefulness": <1-5>,\n'
     '  "verdict": "<one sentence overall assessment>"\n'
     '}}'),
    ("human",
     "Hypothesis: {statement}\n\n"
     "Supporting papers: {supported_by}\n\n"
     "Topic: {topic}"),
])


def judge_scientific_plausibility(
    statement: str,
    supported_by: list[str],
    topic: str,
) -> dict[str, Any]:
    """
    Score a generated hypothesis on 6 scientific quality dimensions.

    Each dimension is scored 1-5 by the LLM. The average is computed
    and returned alongside individual scores and a one-sentence verdict.

    Used in two places:
        - Pipeline (tools.py): runs after originality scoring, result
          stored in hypothesis dict and shown to user in the app.
        - Evaluation (evaluate.py): runs as part of the benchmark,
          individual dimension scores stored in results JSON.

    Args:
        statement: the hypothesis text.
        supported_by: list of paper IDs cited as supporting evidence.
        topic: the user's original research topic.

    Returns:
        Dict with keys: novelty, testability, mechanistic_coherence,
        citation_traceability, conflict_awareness, usefulness,
        verdict, average_score.
    """
    llm = _get_llm(temperature=0.0)
    chain = PLAUSIBILITY_PROMPT | llm | StrOutputParser()

    try:
        input_text = f"Hypothesis: {statement[:200]}..."
        raw = chain.invoke({
            "statement": statement,
            "supported_by": ", ".join(supported_by) if supported_by else "none",
            "topic": topic,
        })
        _log_llm_call(
            config.MAIN_LLM_MODEL, "llm",
            f"plausibility judge: {statement[:40]}",
            input_text, raw,
        )

        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        scores = json.loads(cleaned)

        dimensions = [
            "novelty", "testability", "mechanistic_coherence",
            "citation_traceability", "conflict_awareness", "usefulness",
        ]
        values = [float(scores.get(d, 3)) for d in dimensions]
        average = round(sum(values) / len(values), 2)
        scores["average_score"] = average

        logger.info(
            f"Plausibility judge: avg={average:.2f} | "
            + " | ".join(f"{d}={scores.get(d)}" for d in dimensions)
        )
        return scores

    except Exception as e:
        logger.error(f"Plausibility judge failed: {e}")
        return {
            "novelty": 3, "testability": 3, "mechanistic_coherence": 3,
            "citation_traceability": 3, "conflict_awareness": 3, "usefulness": 3,
            "verdict": "Plausibility scoring unavailable.",
            "average_score": 3.0,
        }
