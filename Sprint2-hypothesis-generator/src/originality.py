"""
Originality checking and gap filtering via cosine similarity.

Two functions:
    1. Filter 1 (pre-generation): past-future gap filter
       Compares past-tagged chunks against future-tagged chunks.
       Only pairs with similarity < threshold are kept as genuine gaps.

    2. Filter 2 (post-generation): originality metric
       Compares each generated hypothesis against all existing paper
       hypotheses. Produces an originality score per hypothesis.

All computations use cosine similarity on embeddings from the same
model used throughout the pipeline (text-embedding-3-small). No LLM
calls are made — this is pure vector math.

Public API:
    - filter_genuine_gaps(past_chunks, future_chunks, embeddings) -> list[GapPair]
    - score_originality(generated_hypotheses, existing_hypotheses, embeddings) -> list[OriginalityResult]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from langchain_openai import OpenAIEmbeddings

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class GapPair:
    """A validated past-future pair representing a genuine research gap."""
    past_text: str
    past_paper_id: str
    future_text: str
    future_paper_id: str
    similarity: float      # Cosine similarity between past and future
    gap_strength: float     # 1 - similarity (higher = bigger gap = more novel potential)


@dataclass
class OriginalityResult:
    """Originality assessment for one generated hypothesis."""
    hypothesis_id: str
    hypothesis_text: str
    max_similarity: float           # Highest similarity to any existing hypothesis
    originality_score: float        # 1 - max_similarity (higher = more original)
    most_similar_to: str            # The existing hypothesis it's most similar to
    is_original: bool               # True if originality_score > threshold
    grade: str = "moderate"         # "very" | "moderate" | "less" (three-category grade)
    grade_label: str = ""           # Human-readable label for display
    grade_color: str = ""           # Hex color for the green-tone palette


# =============================================================================
# Cosine similarity computation
# =============================================================================

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a value between -1 and 1, where 1 means identical direction
    and 0 means orthogonal (unrelated). For normalised embeddings from
    OpenAI, the range is typically 0 to 1.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _batch_embed(texts: list[str], embeddings: OpenAIEmbeddings) -> list[list[float]]:
    """
    Embed a list of texts in one batched API call.

    Wraps the OpenAI embeddings client. Cost is negligible — a few
    hundred tokens at $0.02/1M tokens.
    """
    if not texts:
        return []
    return embeddings.embed_documents(texts)


# =============================================================================
# Filter 1: Past-Future gap filter (pre-generation)
# =============================================================================

def filter_genuine_gaps(
    past_chunks: list[dict],
    future_chunks: list[dict],
    embeddings: OpenAIEmbeddings,
    threshold: float = config.GAP_SIMILARITY_THRESHOLD,
) -> list[GapPair]:
    """
    Compare past-tagged chunks against future-tagged chunks and return
    only the pairs that represent genuine research gaps.

    A pair is considered a genuine gap when the cosine similarity between
    the past chunk and the future chunk is BELOW the threshold. High
    similarity means the future recommendation is just restating what
    was already done — not a real gap.

    Args:
        past_chunks: list of dicts with 'text' and 'paper_id' keys.
        future_chunks: list of dicts with 'text' and 'paper_id' keys.
        embeddings: the OpenAI embeddings client.
        threshold: maximum similarity for a pair to be considered a gap.

    Returns:
        List of GapPair objects sorted by gap_strength (strongest first).
    """
    if not past_chunks or not future_chunks:
        logger.warning("No past or future chunks available for gap filtering")
        return []

    # Embed all texts in two batched calls
    past_texts = [c["text"] for c in past_chunks]
    future_texts = [c["text"] for c in future_chunks]

    past_vecs = _batch_embed(past_texts, embeddings)
    future_vecs = _batch_embed(future_texts, embeddings)

    # Compare each past chunk against each future chunk
    gap_pairs: list[GapPair] = []

    for i, (p_vec, p_chunk) in enumerate(zip(past_vecs, past_chunks)):
        for j, (f_vec, f_chunk) in enumerate(zip(future_vecs, future_chunks)):
            sim = _cosine_similarity(p_vec, f_vec)

            if sim < threshold:
                gap_pairs.append(GapPair(
                    past_text=p_chunk["text"],
                    past_paper_id=p_chunk["paper_id"],
                    future_text=f_chunk["text"],
                    future_paper_id=f_chunk["paper_id"],
                    similarity=sim,
                    gap_strength=1.0 - sim,
                ))

    # Sort by gap strength (biggest gaps first — most potential for novelty)
    gap_pairs.sort(key=lambda g: g.gap_strength, reverse=True)

    logger.info(
        f"Gap filter: {len(past_chunks)} past × {len(future_chunks)} future "
        f"= {len(past_chunks) * len(future_chunks)} pairs checked, "
        f"{len(gap_pairs)} genuine gaps found (threshold={threshold})"
    )

    return gap_pairs


# =============================================================================
# Three-category grading (shared by originality and literature gap scoring)
# =============================================================================

def grade_similarity(similarity: float, context: str = "originality") -> dict[str, str]:
    """
    Convert a cosine similarity value into a three-category grade.

    Same thresholds apply to both originality (hypothesis vs past summary)
    and literature gap (past summary vs future summary) — only the labels
    differ based on context.

    Args:
        similarity: cosine similarity (0 to 1).
        context: "originality" or "gap" — controls the returned labels.

    Returns:
        Dict with 'grade' (very/moderate/less), 'label', and 'color'.
    """
    if context == "gap":
        labels = {
            "very":     "Strong gap",
            "moderate": "Moderate gap",
            "less":     "Weak gap",
        }
    else:  # originality
        labels = {
            "very":     "Very original",
            "moderate": "Moderately original",
            "less":     "Less original",
        }

    if similarity <= config.VERY_ORIGINAL_THRESHOLD:
        grade = "very"
    elif similarity >= config.LESS_ORIGINAL_THRESHOLD:
        grade = "less"
    else:
        grade = "moderate"

    return {
        "grade": grade,
        "label": labels[grade],
        "color": config.BLUE_GRADE_COLORS[grade],
    }


# =============================================================================
# Filter 2: Originality metric — single hypothesis vs. past summary
# =============================================================================

def score_originality_against_summary(
    generated_hypotheses: list[dict],
    past_summary: str,
    embeddings: OpenAIEmbeddings,
) -> list[OriginalityResult]:
    """
    Score originality of generated hypotheses against a consolidated
    past_summary string (instead of individual paper hypotheses).

    The past_summary contains the combined past-tagged content across
    all papers in the library. Comparing against this summary catches
    restatements of ANY past content, not just paper-level hypotheses.

        originality_score = 1 - cos(hypothesis, past_summary)

    Args:
        generated_hypotheses: list of dicts with 'id' and 'statement' keys.
        past_summary: a single string summarising past work across all papers.
        embeddings: the OpenAI embeddings client.

    Returns:
        List of OriginalityResult objects, one per generated hypothesis,
        each annotated with a three-category grade.
    """
    if not generated_hypotheses:
        return []

    if not past_summary or not past_summary.strip():
        logger.info("No past_summary provided — all hypotheses scored as fully original")
        return [
            OriginalityResult(
                hypothesis_id=h.get("id", "H?"),
                hypothesis_text=h.get("statement", ""),
                max_similarity=0.0,
                originality_score=1.0,
                most_similar_to="(no past summary available)",
                is_original=True,
                **{
                    k: v for k, v in
                    zip(
                        ["grade", "grade_label", "grade_color"],
                        [
                            grade_similarity(0.0, "originality")["grade"],
                            grade_similarity(0.0, "originality")["label"],
                            grade_similarity(0.0, "originality")["color"],
                        ],
                    )
                },
            )
            for h in generated_hypotheses
        ]

    # Embed generated hypotheses + past summary (one batched call each)
    gen_texts = [h.get("statement", "") for h in generated_hypotheses]
    gen_vecs = _batch_embed(gen_texts, embeddings)
    past_vec = _batch_embed([past_summary], embeddings)[0]

    results: list[OriginalityResult] = []

    for g_vec, g_hyp in zip(gen_vecs, generated_hypotheses):
        sim = _cosine_similarity(g_vec, past_vec)
        originality = 1.0 - sim
        grade_info = grade_similarity(sim, context="originality")

        results.append(OriginalityResult(
            hypothesis_id=g_hyp.get("id", "H?"),
            hypothesis_text=g_hyp.get("statement", ""),
            max_similarity=sim,
            originality_score=originality,
            most_similar_to="(past summary)",
            is_original=(sim < config.ORIGINALITY_THRESHOLD),
            grade=grade_info["grade"],
            grade_label=grade_info["label"],
            grade_color=grade_info["color"],
        ))

        logger.info(
            f"{g_hyp.get('id', 'H?')}: originality={originality:.3f} "
            f"(sim to past_summary={sim:.3f}, grade={grade_info['label']})"
        )

    return results


# =============================================================================
# Filter 2: Originality metric (post-generation)
# =============================================================================

def score_originality(
    generated_hypotheses: list[dict],
    existing_hypotheses: list[str],
    embeddings: OpenAIEmbeddings,
    threshold: float = config.ORIGINALITY_THRESHOLD,
) -> list[OriginalityResult]:
    """
    Score how original each generated hypothesis is compared to existing
    paper hypotheses.

    For each generated hypothesis, cosine similarity is computed against
    every existing hypothesis. The originality score is:
        originality = 1 - max(similarity to any existing hypothesis)

    A high originality score means the generated hypothesis is semantically
    distant from everything that has already been tested — genuinely novel.

    Args:
        generated_hypotheses: list of dicts with 'id' and 'statement' keys.
        existing_hypotheses: list of hypothesis strings from paper metadata.
        embeddings: the OpenAI embeddings client.
        threshold: similarity above this flags a hypothesis as unoriginal.

    Returns:
        List of OriginalityResult objects, one per generated hypothesis.
    """
    if not generated_hypotheses:
        return []

    if not existing_hypotheses:
        # If no existing hypotheses to compare against, everything is original
        logger.info("No existing hypotheses to compare — all scored as original")
        return [
            OriginalityResult(
                hypothesis_id=h.get("id", "H?"),
                hypothesis_text=h.get("statement", ""),
                max_similarity=0.0,
                originality_score=1.0,
                most_similar_to="(no existing hypotheses to compare)",
                is_original=True,
            )
            for h in generated_hypotheses
        ]

    # Embed generated and existing hypotheses
    gen_texts = [h.get("statement", "") for h in generated_hypotheses]
    gen_vecs = _batch_embed(gen_texts, embeddings)
    exist_vecs = _batch_embed(existing_hypotheses, embeddings)

    results: list[OriginalityResult] = []

    for i, (g_vec, g_hyp) in enumerate(zip(gen_vecs, generated_hypotheses)):
        max_sim = 0.0
        most_similar = ""

        for j, (e_vec, e_text) in enumerate(zip(exist_vecs, existing_hypotheses)):
            sim = _cosine_similarity(g_vec, e_vec)
            if sim > max_sim:
                max_sim = sim
                most_similar = e_text

        originality = 1.0 - max_sim

        results.append(OriginalityResult(
            hypothesis_id=g_hyp.get("id", "H?"),
            hypothesis_text=g_hyp.get("statement", ""),
            max_similarity=max_sim,
            originality_score=originality,
            most_similar_to=most_similar,
            is_original=(max_sim < threshold),
        ))

        logger.info(
            f"{g_hyp.get('id', 'H?')}: originality={originality:.3f} "
            f"(max_sim={max_sim:.3f}, {'✅ original' if max_sim < threshold else '⚠️ too similar'})"
        )

    return results
