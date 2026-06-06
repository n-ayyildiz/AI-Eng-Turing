"""
Originality scoring for Neurohypothesis v2.

Ported from v1 originality.py with two changes:
    1. cosine_similarity and grade_similarity imported from src.utils
       (single source of truth — no duplication).
    2. loguru replaces stdlib logging.

Two public functions:
    score_originality_against_summary — used in N14 (score_originality):
        compares a generated hypothesis against the past_summary embedding.
        originality_score = 1 − cosine(hypothesis, past_summary)

    filter_genuine_gaps — used in N12 (compute_gap):
        compares past-tagged chunk embeddings against future-tagged chunk
        embeddings to find pairs with a genuine semantic gap.
        gap_score = 1 − cosine(past_chunk, future_chunk)

Public API:
    - OriginalityResult     dataclass
    - GapPair               dataclass
    - score_originality_against_summary(generated, past_summary, embeddings)
      -> list[OriginalityResult]
    - filter_genuine_gaps(past_chunks, future_chunks, embeddings)
      -> list[GapPair]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from src.utils import cosine_similarity, embed_texts, grade_similarity

import config

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class OriginalityResult:
    """Originality assessment for one generated hypothesis."""
    hypothesis_id:   str
    hypothesis_text: str
    similarity:      float    # cosine(hypothesis, past_summary)
    originality_score: float  # 1 - similarity
    grade:           str      # "very" | "moderate" | "less"
    grade_label:     str      # human-readable label
    grade_color:     str      # hex colour for UI
    passes_gate:     bool     # originality_score >= ORIGINALITY_PASS_THRESHOLD


@dataclass
class GapPair:
    """A validated past-future pair representing a genuine research gap."""
    past_text:     str
    past_paper_id: str
    future_text:   str
    future_paper_id: str
    similarity:    float   # cosine(past, future)
    gap_score:     float   # 1 - similarity (higher = bigger gap)
    past_idx:      int = -1   # index into past_vecs (for MMR pair selection)
    future_idx:    int = -1   # index into future_vecs


# =============================================================================
# N14 — Score originality of a hypothesis against the past summary
# =============================================================================

def score_originality_against_summary(
    generated_hypotheses: list[dict],
    past_summary:         str,
    embeddings:           OpenAIEmbeddings,
) -> list[OriginalityResult]:
    """
    Score how original each generated hypothesis is compared to the
    consolidated past_summary (the 3-bullet summary of past-tagged content).

    Formula:  originality_score = 1 − cosine(hypothesis_embedding, past_summary_embedding)

    A high score means the hypothesis is semantically distant from past
    work — genuinely novel.  A low score means it likely restates something
    already established.

    Args:
        generated_hypotheses: list of dicts, each with 'id' and 'statement'.
        past_summary:         the past_summary string produced by N10.
        embeddings:           the shared OpenAIEmbeddings client.

    Returns:
        List of OriginalityResult, one per hypothesis, with grade and gate flag.
    """
    if not generated_hypotheses:
        return []

    if not past_summary or not past_summary.strip():
        logger.info("No past_summary — all hypotheses scored as fully original")
        return [
            _make_result(h, similarity=0.0)
            for h in generated_hypotheses
        ]

    # Batch embed all hypothesis texts + the past summary
    hyp_texts  = [h.get("statement", "") for h in generated_hypotheses]
    all_vecs   = embed_texts(hyp_texts + [past_summary])
    hyp_vecs   = all_vecs[:-1]
    past_vec   = all_vecs[-1]

    results: list[OriginalityResult] = []
    for hyp, vec in zip(generated_hypotheses, hyp_vecs):
        sim = cosine_similarity(vec, past_vec)
        results.append(_make_result(hyp, similarity=sim))
        logger.debug(
            f"{hyp.get('id', 'H?')}: "
            f"originality={1-sim:.3f} sim={sim:.3f} "
            f"grade={grade_similarity(sim)['grade']}"
        )

    return results


def _make_result(hyp: dict, similarity: float) -> OriginalityResult:
    """Build an OriginalityResult from a hypothesis dict and a similarity score."""
    grade_info = grade_similarity(similarity, context="originality")
    return OriginalityResult(
        hypothesis_id=hyp.get("id", "H?"),
        hypothesis_text=hyp.get("statement", ""),
        similarity=similarity,
        originality_score=round(1.0 - similarity, 4),
        grade=grade_info["grade"],
        grade_label=grade_info["label"],
        grade_color=grade_info["color"],
        passes_gate=(1.0 - similarity) >= config.ORIGINALITY_PASS_THRESHOLD,
    )


# =============================================================================
# N12 — Compute gap score between past and future chunk sets
# =============================================================================

def filter_genuine_gaps(
    past_chunks:   list[dict],
    future_chunks: list[dict],
    embeddings:    OpenAIEmbeddings,
    threshold:     float = config.LESS_ORIGINAL_THRESHOLD,
) -> list[GapPair]:
    """
    Compare past-tagged chunks against future-tagged chunks and return
    only the pairs that represent a genuine research gap.

    A pair is a genuine gap when cosine(past, future) < threshold —
    the future recommendation is semantically distant from what was done,
    meaning there is real novel ground between them.

    Args:
        past_chunks:   list of dicts with 'text' and 'paper_id'.
        future_chunks: list of dicts with 'text' and 'paper_id'.
        embeddings:    the shared OpenAIEmbeddings client.
        threshold:     similarity ceiling for a pair to qualify as a gap.

    Returns:
        List of GapPair sorted by gap_score descending (strongest gaps first).
    """
    if not past_chunks or not future_chunks:
        logger.warning("filter_genuine_gaps: empty past or future chunk list")
        return []

    past_vecs   = embed_texts([c["text"] for c in past_chunks])
    future_vecs = embed_texts([c["text"] for c in future_chunks])

    gaps: list[GapPair] = []
    for p_vec, p_chunk in zip(past_vecs, past_chunks):
        for f_vec, f_chunk in zip(future_vecs, future_chunks):
            sim = cosine_similarity(p_vec, f_vec)
            if sim < threshold:
                gaps.append(GapPair(
                    past_text=p_chunk["text"],
                    past_paper_id=p_chunk["paper_id"],
                    future_text=f_chunk["text"],
                    future_paper_id=f_chunk["paper_id"],
                    similarity=sim,
                    gap_score=round(1.0 - sim, 4),
                ))

    gaps.sort(key=lambda g: g.gap_score, reverse=True)
    logger.info(
        f"Gap filter: {len(past_chunks)}×{len(future_chunks)} pairs checked → "
        f"{len(gaps)} genuine gaps (threshold={threshold})"
    )
    return gaps


# =============================================================================
# Path A — band pairing + tiered/MMR selection + diversity gate
# =============================================================================

def build_gap_pairs(
    past_items:   list[dict],
    future_items: list[dict],
    band:         tuple[float, float] = (config.GAP_BAND_LOW, config.GAP_BAND_HIGH),
) -> tuple[list[GapPair], list[list[float]], list[list[float]]]:
    """
    Build past->future pairs in a RELEVANCE BAND: similar enough to share a
    research thread, divergent enough to be a real gap.  Pairs outside the band
    are dropped (too high = restatement, too low = unrelated).

    Returns (pairs sorted by gap_score desc, past_vecs, future_vecs).
    GapPair.past_idx / future_idx index into those vector lists (for MMR).
    """
    if not past_items or not future_items:
        return [], [], []

    past_vecs   = embed_texts([c["text"] for c in past_items])
    future_vecs = embed_texts([c["text"] for c in future_items])
    low, high = band

    pairs: list[GapPair] = []
    for i, (pv, pc) in enumerate(zip(past_vecs, past_items)):
        for j, (fv, fc) in enumerate(zip(future_vecs, future_items)):
            sim = cosine_similarity(pv, fv)
            if low <= sim <= high:
                pairs.append(GapPair(
                    past_text=pc["text"],     past_paper_id=pc.get("paper_id", ""),
                    future_text=fc["text"],   future_paper_id=fc.get("paper_id", ""),
                    similarity=sim,           gap_score=round(1.0 - sim, 4),
                    past_idx=i,               future_idx=j,
                ))
    pairs.sort(key=lambda g: g.gap_score, reverse=True)
    logger.info(
        f"build_gap_pairs: {len(past_items)}×{len(future_items)} → "
        f"{len(pairs)} in-band pairs [{low}, {high}]"
    )
    return pairs, past_vecs, future_vecs


def select_pairs_tiered_mmr(
    pairs:       list[GapPair],
    past_vecs:   list[list[float]],
    future_vecs: list[list[float]],
    n:           int,
    mmr_lambda:  float = config.GAP_MMR_LAMBDA,
) -> list[GapPair]:
    """
    Select up to n pairs maximizing diversity, in priority tiers:
      Tier 1: neither past nor future index reused (most distinct)
      Tier 2: exactly one of the two reused
      Tier 3: any remaining combination
    Within the best available tier, MMR picks the pair that is high-gap AND
    least similar (pair-embedding) to pairs already chosen.
    """
    if not pairs:
        return []

    def pair_vec(p: GapPair) -> list[float]:
        a = np.array(past_vecs[p.past_idx]); b = np.array(future_vecs[p.future_idx])
        return ((a + b) / 2.0).tolist()

    selected: list[GapPair] = []
    sel_vecs: list[list[float]] = []
    used_past: set[int] = set()
    used_future: set[int] = set()
    remaining = list(pairs)

    def tier_of(p: GapPair) -> int:
        pu, fu = p.past_idx in used_past, p.future_idx in used_future
        if not pu and not fu:
            return 1
        if pu ^ fu:
            return 2
        return 3

    while remaining and len(selected) < n:
        best_tier = min(tier_of(p) for p in remaining)
        cands = [p for p in remaining if tier_of(p) == best_tier]  # gap-desc order preserved
        if not sel_vecs:
            choice = cands[0]
        else:
            def mmr(p: GapPair) -> float:
                pv = pair_vec(p)
                max_sim = max(cosine_similarity(pv, sv) for sv in sel_vecs)
                return mmr_lambda * p.gap_score - (1 - mmr_lambda) * max_sim
            choice = max(cands, key=mmr)
        selected.append(choice)
        sel_vecs.append(pair_vec(choice))
        used_past.add(choice.past_idx)
        used_future.add(choice.future_idx)
        remaining.remove(choice)

    logger.info(f"select_pairs_tiered_mmr: chose {len(selected)} of {len(pairs)} pairs (n={n})")
    return selected


def max_similarity_to_existing(text: str, existing_texts: list[str]) -> float:
    """Diversity gate: highest cosine of `text` against any previously accepted text."""
    if not existing_texts or not text or not text.strip():
        return 0.0
    vecs = embed_texts([text] + existing_texts)
    return max(cosine_similarity(vecs[0], v) for v in vecs[1:])
