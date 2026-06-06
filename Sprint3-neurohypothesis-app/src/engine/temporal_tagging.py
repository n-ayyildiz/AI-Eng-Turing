"""
Temporal tagging of text chunks: past research vs future recommendations.

Each chunk is classified as "past", "future", or "neutral" using
cosine similarity against two reference descriptions from config.
Neutral chunks (similarity scores too close to call) are resolved by
an optional LLM fallback call.

This module is called once during ingestion (N4a ingest_pdfs) and again
for PubMed abstract chunks (N5h extract_metadata_pubmed).

Ported from v1 ingest.py (_tag_chunks_past_future) and
generate.py (classify_neutral_chunk) with loguru and utils imports.

Public API:
    - tag_chunks_past_future(chunks, embeddings) -> list[Document]
    - classify_neutral_chunk_llm(chunk_text, node_name) -> str
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from src.utils import cosine_similarity, embed_texts

import config

# =============================================================================
# Reference vector cache
# =============================================================================

_reference_vecs: dict[str, list[float]] | None = None


def _get_reference_vecs(embeddings: OpenAIEmbeddings) -> dict[str, list[float]]:
    """
    Embed the past/future reference descriptions (once, then cached).

    The two reference strings are defined in config.TEMPORAL_REFERENCES.
    They describe what "past research" vs "future directions" language
    looks like, and are used as anchors for cosine similarity tagging.
    """
    global _reference_vecs
    if _reference_vecs is None:
        refs   = config.TEMPORAL_REFERENCES
        vecs   = embed_texts([refs["past"], refs["future"]])
        _reference_vecs = {"past": vecs[0], "future": vecs[1]}
        logger.debug("Temporal reference vectors cached")
    return _reference_vecs


# =============================================================================
# Core tagging logic
# =============================================================================

def _tag_one_chunk(
    chunk_text: str,
    ref_vecs:   dict[str, list[float]],
    chunk_vec:  list[float],
) -> tuple[str, float, float]:
    """
    Classify one chunk as past / future / neutral via cosine similarity.

    Returns:
        (tag, past_sim, future_sim)
    """
    past_sim   = cosine_similarity(chunk_vec, ref_vecs["past"])
    future_sim = cosine_similarity(chunk_vec, ref_vecs["future"])
    margin     = config.TEMPORAL_NEUTRAL_MARGIN

    if abs(past_sim - future_sim) < margin:
        tag = "neutral"
    elif past_sim >= future_sim:
        tag = "past"
    else:
        tag = "future"

    return tag, past_sim, future_sim


def tag_chunks_past_future(
    chunks:     list[Document],
    embeddings: OpenAIEmbeddings,
) -> list[Document]:
    """
    Classify every chunk as "past", "future", or "neutral" and store the
    result in chunk.metadata["temporal_lean"].

    All chunks are embedded in a single batched API call for efficiency.
    The cost is ~$0.002 for 300 chunks at text-embedding-3-small pricing.

    Neutral chunks remain tagged "neutral" in metadata — the hypothesis
    engine uses past and future chunks; neutral ones contribute context
    but are not directly used in the gap analysis.

    Args:
        chunks:     list of Document objects to tag (metadata updated in-place).
        embeddings: the shared OpenAIEmbeddings client.

    Returns:
        The same list with temporal_lean metadata filled in.
    """
    if not chunks:
        return chunks

    ref_vecs = _get_reference_vecs(embeddings)

    # Batch-embed all chunk texts in one API call
    texts      = [c.page_content for c in chunks]
    chunk_vecs = embed_texts(texts)

    past_count = future_count = neutral_count = 0

    for chunk, vec in zip(chunks, chunk_vecs):
        tag, past_sim, future_sim = _tag_one_chunk(chunk.page_content, ref_vecs, vec)
        chunk.metadata["temporal_lean"]   = tag
        chunk.metadata["past_similarity"] = round(past_sim,   4)
        chunk.metadata["future_similarity"] = round(future_sim, 4)

        if tag == "past":
            past_count   += 1
        elif tag == "future":
            future_count += 1
        else:
            neutral_count += 1

    logger.info(
        f"Temporal tagging: {len(chunks)} chunks → "
        f"past={past_count} future={future_count} neutral={neutral_count}"
    )
    return chunks


# =============================================================================
# LLM fallback for neutral chunks (called selectively, not in batch)
# =============================================================================

def classify_neutral_chunk_llm(
    chunk_text: str,
    node_name:  str = "temporal_tagging",
) -> str:
    """
    Ask the LLM to classify a single neutral-tagged chunk as past/future/neutral.

    Used as a secondary pass: after batch embedding, any chunk tagged
    "neutral" (both similarities too close to call) can optionally be
    reclassified by an LLM call. This improves recall on ambiguous
    paragraphs that use both past and future tense.

    Kept lightweight: temperature=0, forced to return exactly one word.

    Args:
        chunk_text: the chunk content to classify.
        node_name:  caller label for cost tracking.

    Returns:
        "past", "future", or "neutral".
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from src.cost_tracking import count_tokens, get_tracker

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You classify neuroscience paper text as representing PAST research "
         "findings or FUTURE research recommendations.\n\n"
         "PAST = described findings, established results, observations, "
         "what has been studied, methods used, completed investigations.\n"
         "FUTURE = limitations of the present work, recommendations for future "
         "research, unresolved questions, suggested investigations.\n\n"
         "If the chunk is a genuine mix with no clear lean, return 'neutral'.\n"
         "Return ONLY one word: past, future, or neutral. No punctuation."),
        ("human", "Chunk:\n{chunk_text}"),
    ])

    llm   = ChatOpenAI(model=config.MAIN_LLM_MODEL, temperature=0.0, seed=config.LLM_SEED)
    chain = prompt | llm | StrOutputParser()

    try:
        input_text = f"Chunk:\n{chunk_text[:300]}"
        raw        = chain.invoke({"chunk_text": chunk_text})

        # Log to cost tracker
        tracker = get_tracker()
        tracker.log_call(
            node_name=node_name,
            model=config.MAIN_LLM_MODEL,
            call_type="llm",
            summary="classify neutral chunk",
            input_tokens=count_tokens(input_text, config.MAIN_LLM_MODEL),
            output_tokens=count_tokens(raw,        config.MAIN_LLM_MODEL),
        )

        answer = raw.strip().lower().strip(".,;:'\"")
        if answer in ("past", "future", "neutral"):
            return answer
        logger.warning(f"Unexpected classification: {raw!r} — defaulting to neutral")
        return "neutral"

    except Exception as exc:
        logger.error(f"LLM neutral chunk classification failed: {exc}")
        return "neutral"
