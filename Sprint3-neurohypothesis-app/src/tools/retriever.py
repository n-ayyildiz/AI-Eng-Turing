"""
Hybrid retrieval for Neurohypothesis v2: semantic search (ChromaDB) + BM25 + RRF fusion.

Ported from v1 retrievers.py with two changes:
    1. Per-session Chroma collection — callers pass the vectorstore object
       (which already targets the right per-session collection); no global
       collection name is used here.
    2. loguru replaces stdlib logging throughout.

Why hybrid:
    Semantic search (ChromaDB) — cosine similarity on OpenAI embeddings.
      Good for paraphrased concepts and meaning-level matching.
    BM25 keyword search — exact term matching on raw chunk text.
      Critical for neuroscience jargon (e.g. "VBM", "amyloid-beta",
      "APOE-ε4") that semantic embeddings may compress or miss.
    Reciprocal Rank Fusion (RRF) — merges both ranked lists. Fully local,
      no cloud dependency, consistently outperforms individual rankers.

Public API:
    - ScoredChunk           dataclass
    - get_all_chunks(vectorstore, section_filter) -> list[Document]
    - semantic_search(query, vectorstore, k, section_filter) -> list[ScoredChunk]
    - bm25_search(query, all_chunks, k) -> list[ScoredChunk]
    - hybrid_search(query, vectorstore, k, section_filter) -> list[ScoredChunk]
    - retrieve_by_temporal_tag(tag, vectorstore, k) -> list[ScoredChunk]
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

import config

# =============================================================================
# Data structures
# =============================================================================


@dataclass
class ScoredChunk:
    """
    A retrieved chunk annotated with all three retrieval scores.

    Displayed in the developer panel so the source and ranking of each
    context chunk is visible and auditable.
    """

    document: Document
    semantic_score: float = 0.0  # cosine similarity [0, 1]
    bm25_score: float = 0.0  # raw BM25 (unbounded, higher = better)
    rrf_score: float = 0.0  # fused score after RRF

    @property
    def paper_id(self) -> str:
        return self.document.metadata.get("paper_id", "unknown")

    @property
    def section_type(self) -> str:
        return self.document.metadata.get("section_type", "unknown")

    @property
    def temporal_lean(self) -> str:
        return self.document.metadata.get("temporal_lean", "neutral")

    @property
    def text(self) -> str:
        return self.document.page_content


# =============================================================================
# Chunk fetch (for BM25 input and temporal-tag filtering)
# =============================================================================


def get_all_chunks(
    vectorstore: Chroma,
    section_filter: str | None = None,
    temporal_filter: str | None = None,
    categories: list[str] | None = None,
) -> list[Document]:
    """
    Fetch all stored chunks from Chroma as Document objects.

    Args:
        vectorstore:     the per-session Chroma vectorstore.
        section_filter:  restrict to one section_type if set.
        temporal_filter: restrict to one temporal_lean tag if set.
        categories:      restrict to one or more category names if set.
    """
    conditions: list[dict] = []
    if section_filter:
        conditions.append({"section_type": {"$eq": section_filter}})
    if temporal_filter:
        conditions.append({"temporal_lean": {"$eq": temporal_filter}})
    if categories:
        if len(categories) == 1:
            conditions.append({"category": {"$eq": categories[0]}})
        else:
            conditions.append({"category": {"$in": categories}})

    if not conditions:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    try:
        result = vectorstore.get(where=where)
    except Exception as exc:
        logger.error(f"get_all_chunks failed: {exc}")
        return []

    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    return [Document(page_content=text, metadata=meta) for text, meta in zip(docs, metas)]


# =============================================================================
# Semantic search
# =============================================================================


def semantic_search(
    query: str,
    vectorstore: Chroma,
    k: int = config.SEMANTIC_TOP_K,
    section_filter: str | None = None,
    categories: list[str] | None = None,
) -> list[ScoredChunk]:
    """
    Retrieve the top-k most semantically similar chunks via ChromaDB.

    Args:
        query:          the search string.
        vectorstore:    the per-session Chroma vectorstore.
        k:              number of results to return.
        section_filter: restrict to one section_type if set.
        categories:     restrict to one or more category names if set.
    """
    conditions: list[dict] = []
    if section_filter:
        conditions.append({"section_type": {"$eq": section_filter}})
    if categories:
        if len(categories) == 1:
            conditions.append({"category": {"$eq": categories[0]}})
        else:
            conditions.append({"category": {"$in": categories}})

    if not conditions:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=where,
        )
    except Exception as exc:
        logger.error(f"Semantic search failed for '{query[:50]}': {exc}")
        return []

    return [ScoredChunk(document=doc, semantic_score=score) for doc, score in results]


# =============================================================================
# BM25 keyword search
# =============================================================================


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def bm25_search(
    query: str,
    all_chunks: list[Document],
    k: int = config.BM25_TOP_K,
) -> list[ScoredChunk]:
    """
    Rank all chunks by BM25 keyword relevance to the query.

    The BM25 index is rebuilt per call because the corpus can grow
    during a session (local PDFs → PubMed results merged). For ≤ 500
    chunks this takes < 10 ms — negligible.

    Args:
        query:      the search string.
        all_chunks: full corpus to rank (pre-fetched by get_all_chunks).
        k:          number of top results to return.

    Returns:
        List of ScoredChunk sorted by BM25 score (highest first).
    """
    if not all_chunks:
        return []

    corpus_tokens = [_tokenize(doc.page_content) for doc in all_chunks]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(_tokenize(query))

    scored = [
        ScoredChunk(document=doc, bm25_score=float(score)) for doc, score in zip(all_chunks, scores)
    ]
    scored.sort(key=lambda x: x.bm25_score, reverse=True)
    return scored[:k]


# =============================================================================
# Reciprocal Rank Fusion
# =============================================================================


def _reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    k: int = config.RRF_K,
) -> list[ScoredChunk]:
    """
    Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    RRF score(doc) = Σ 1 / (k + rank_i) over all lists that contain doc,
    where rank is 1-indexed and k=60 is the Cormack et al. 2009 constant.

    Deduplication key: chunk page_content (same text from two retrievers
    should count once with scores merged, not appear twice).
    """
    fused: dict[str, tuple[float, ScoredChunk]] = {}

    for ranked_list in ranked_lists:
        for rank_idx, sc in enumerate(ranked_list):
            key = sc.document.page_content
            contribution = 1.0 / (k + rank_idx + 1)

            if key in fused:
                prev_score, prev_sc = fused[key]
                fused[key] = (
                    prev_score + contribution,
                    ScoredChunk(
                        document=prev_sc.document,
                        semantic_score=max(prev_sc.semantic_score, sc.semantic_score),
                        bm25_score=max(prev_sc.bm25_score, sc.bm25_score),
                        rrf_score=prev_score + contribution,
                    ),
                )
            else:
                fused[key] = (
                    contribution,
                    ScoredChunk(
                        document=sc.document,
                        semantic_score=sc.semantic_score,
                        bm25_score=sc.bm25_score,
                        rrf_score=contribution,
                    ),
                )

    results = [sc for _, sc in fused.values()]
    results.sort(key=lambda x: x.rrf_score, reverse=True)
    return results


# =============================================================================
# Hybrid search (main entry point)
# =============================================================================


def hybrid_search(
    query: str,
    vectorstore: Chroma,
    k: int = config.FINAL_TOP_K,
    section_filter: str | None = None,
) -> list[ScoredChunk]:
    """
    Run semantic + BM25 search in parallel then fuse via RRF.

    This is the primary retrieval function called by all nodes that need
    grounded context (N4b retrieve_local, N5g pubmed_search_alt, etc.).

    Args:
        query:          the search string.
        vectorstore:    the per-session Chroma vectorstore.
        k:              final result count after fusion.
        section_filter: restrict both retrievers to one section type.

    Returns:
        List of ScoredChunk sorted by fused RRF score (highest first),
        capped at k. Each chunk carries all three scores for the dev panel.
    """
    sem_results = semantic_search(
        query, vectorstore, k=config.SEMANTIC_TOP_K, section_filter=section_filter
    )
    all_chunks = get_all_chunks(vectorstore, section_filter=section_filter)
    bm25_results = bm25_search(query, all_chunks, k=config.BM25_TOP_K)
    fused = _reciprocal_rank_fusion([sem_results, bm25_results])

    logger.debug(
        f"hybrid_search '{query[:50]}' — "
        f"sem={len(sem_results)} bm25={len(bm25_results)} "
        f"fused={len(fused)} returned={min(k, len(fused))}"
    )
    return fused[:k]


# =============================================================================
# Temporal-tag retrieval (for N9 tag_past_future)
# =============================================================================

_LIMITATIONS_SEMANTIC_QUERY = (
    "study limitations, methodological constraints, caveats, "
    "shortcomings, future research directions, recommendations "
    "for future work, what remains to be investigated"
)


def retrieve_by_temporal_tag(
    tag: str,
    vectorstore: Chroma,
    k: int = config.FINAL_TOP_K,
    topic: str = "",
    categories: list[str] | None = None,
) -> list[ScoredChunk]:
    """
    Retrieve chunks filtered by temporal_lean metadata tag.

    Args:
        tag:         "past" | "future" | "neutral".
        vectorstore: the per-session Chroma vectorstore.
        k:           result count.
        topic:       user topic, prepended to the query for 'future' pulls.
        categories:  restrict to one or more category names if set.
    """
    if tag == "future" and topic:
        query = f"{topic} — {_LIMITATIONS_SEMANTIC_QUERY}"
    else:
        query = topic or tag

    sem = semantic_search(
        query,
        vectorstore,
        k=config.SEMANTIC_TOP_K,
        section_filter="limitations_future" if tag == "future" else None,
        categories=categories,
    )
    all_chunks = get_all_chunks(vectorstore, temporal_filter=tag, categories=categories)
    bm25_results = bm25_search(query, all_chunks, k=config.BM25_TOP_K)
    fused = _reciprocal_rank_fusion([sem, bm25_results])

    logger.debug(
        f"retrieve_by_temporal_tag tag={tag} topic='{topic[:40]}' "
        f"cats={categories} → {min(k, len(fused))} chunks"
    )
    return fused[:k]
