"""
Hybrid retrieval: semantic search (ChromaDB) + BM25 keyword search + RRF fusion.

This module is the "read side" of the RAG pipeline. Every tool and
generation chain that needs context from the knowledge base calls
into here.

Why hybrid:
    - Semantic search (ChromaDB): cosine similarity on embedded vectors.
      Good for paraphrased concepts and meaning-level matching.
    - BM25 keyword search: exact term matching on raw chunk text.
      Good for domain-specific jargon (e.g. "LDL-cholesterol", "VBM",
      "amyloid-beta", "p-tau") that semantic search might miss because
      the embedding model has limited exposure to specialised terminology.
    - Reciprocal Rank Fusion (RRF): merges both ranked lists into one
      final list. Fully local — no cloud dependency.

Public API:
    - semantic_search(query, vectorstore, k, filter) -> list[ScoredChunk]
    - bm25_search(query, all_chunks, k) -> list[ScoredChunk]
    - hybrid_search(query, vectorstore, k, filter) -> list[ScoredChunk]
    - retrieve_limitations(topic, vectorstore, k) -> list[ScoredChunk]
    - get_all_chunks(vectorstore, filter) -> list[Document]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ScoredChunk:
    """A retrieved chunk with its source scores and metadata.

    Displayed in the Sources tab so the user can see which chunks were
    retrieved, from which paper/section, and how they were ranked.
    """
    document: Document
    semantic_score: float = 0.0   # Cosine similarity (0-1, higher = better)
    bm25_score: float = 0.0      # BM25 raw score (unbounded, higher = better)
    rrf_score: float = 0.0       # Fused score after RRF
    source: str = ""              # "semantic", "bm25", or "hybrid"

    @property
    def paper_id(self) -> str:
        return self.document.metadata.get("paper_id", "unknown")

    @property
    def section_type(self) -> str:
        return self.document.metadata.get("section_type", "unknown")

    @property
    def text(self) -> str:
        return self.document.page_content


# =============================================================================
# Chunk retrieval from ChromaDB (for BM25 input)
# =============================================================================

def get_all_chunks(
    vectorstore: Chroma,
    section_filter: str | None = None,
) -> list[Document]:
    """
    Fetch all stored chunks from ChromaDB as Document objects.

    BM25 needs the full corpus text to build its term-frequency index.
    This function pulls all chunks (optionally filtered by section_type)
    so BM25 can rank them.

    Args:
        vectorstore: the ChromaDB vector store.
        section_filter: if provided, only chunks with this section_type
                        are returned (e.g. "limitations_future").

    Returns:
        List of LangChain Document objects with metadata intact.
    """
    where = {"section_type": section_filter} if section_filter else None

    try:
        result = vectorstore.get(where=where)
    except Exception as e:
        logger.error(f"Failed to fetch chunks from ChromaDB: {e}")
        return []

    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(docs, metas)
    ]


# =============================================================================
# Semantic search (ChromaDB cosine similarity)
# =============================================================================

def semantic_search(
    query: str,
    vectorstore: Chroma,
    k: int = config.SEMANTIC_TOP_K,
    section_filter: str | None = None,
) -> list[ScoredChunk]:
    """
    Retrieve the top-k most semantically similar chunks to the query.

    The query is embedded by the same model used during ingestion
    (text-embedding-3-small) and compared against all stored chunk
    vectors via cosine similarity.

    Args:
        query: the user's search string.
        vectorstore: the ChromaDB vector store.
        k: number of results to return.
        section_filter: if provided, only chunks with this section_type
                        are searched.

    Returns:
        List of ScoredChunk objects sorted by similarity (highest first).
    """
    where = {"section_type": section_filter} if section_filter else None

    try:
        # similarity_search_with_relevance_scores returns (Document, score)
        # tuples where score is cosine similarity in [0, 1].
        results = vectorstore.similarity_search_with_relevance_scores(
            query, k=k, filter=where,
        )
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []

    return [
        ScoredChunk(
            document=doc,
            semantic_score=score,
            source="semantic",
        )
        for doc, score in results
    ]


# =============================================================================
# BM25 keyword search
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercased tokenizer for BM25.

    This is deliberately simple. BM25 works on raw token overlap, so
    keeping the tokenizer basic means domain terms like "LDL-cholesterol",
    "p-tau", and "amyloid-beta" are matched as-is without being broken
    apart by a sophisticated tokenizer.
    """
    return text.lower().split()


def bm25_search(
    query: str,
    all_chunks: list[Document],
    k: int = config.BM25_TOP_K,
) -> list[ScoredChunk]:
    """
    Rank all chunks by BM25 keyword relevance to the query.

    BM25 (Okapi BM25) is a probabilistic ranking function that scores
    documents based on term frequency, inverse document frequency, and
    document length normalisation. It excels at finding exact term
    matches — critical for neuroscience jargon like "VBM", "TBSS",
    "alpha-synuclein", or "LDL-cholesterol" that semantic embeddings
    may not represent well.

    The BM25 index is built fresh for each query because the chunk
    corpus can change during a session (e.g. after PubMed results are
    ingested in Phase 2). For ~300 chunks this takes <10ms — negligible.

    Args:
        query: the user's search string.
        all_chunks: the full list of Document objects to search over.
        k: number of top results to return.

    Returns:
        List of ScoredChunk objects sorted by BM25 score (highest first).
    """
    if not all_chunks:
        return []

    # Build BM25 index from all chunk texts
    corpus_tokens = [_tokenize(doc.page_content) for doc in all_chunks]
    bm25 = BM25Okapi(corpus_tokens)

    # Score the query against the corpus
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Pair each chunk with its BM25 score, sort descending, take top-k
    scored = [
        ScoredChunk(
            document=doc,
            bm25_score=float(score),
            source="bm25",
        )
        for doc, score in zip(all_chunks, scores)
    ]
    scored.sort(key=lambda x: x.bm25_score, reverse=True)

    return scored[:k]


# =============================================================================
# Reciprocal Rank Fusion (RRF)
# =============================================================================

def _reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    k: int = config.RRF_K,
) -> list[ScoredChunk]:
    """
    Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    RRF score for a document = sum over all lists of 1 / (k + rank),
    where rank is 1-indexed and k is a smoothing constant (default 60,
    from the original Cormack et al. 2009 paper).

    This approach is simple, requires no score normalisation between
    different retrieval methods, and consistently outperforms individual
    rankers in practice.

    Args:
        ranked_lists: list of ranked ScoredChunk lists (one per retriever).
        k: smoothing constant. Higher k = more weight to lower-ranked results.

    Returns:
        Merged list sorted by RRF score (highest first).
    """
    # Map from chunk text -> accumulated RRF score + best ScoredChunk object.
    # Using page_content as the dedup key because the same chunk can appear
    # in both the semantic and BM25 results.
    fused: dict[str, tuple[float, ScoredChunk]] = {}

    for ranked_list in ranked_lists:
        for rank_idx, scored_chunk in enumerate(ranked_list):
            content_key = scored_chunk.document.page_content
            rrf_contribution = 1.0 / (k + rank_idx + 1)  # rank is 1-indexed

            if content_key in fused:
                existing_score, existing_chunk = fused[content_key]
                # Accumulate score, keep the chunk with richer score info
                new_score = existing_score + rrf_contribution
                # Merge scores from both retrievers onto the chunk
                merged = ScoredChunk(
                    document=existing_chunk.document,
                    semantic_score=max(existing_chunk.semantic_score, scored_chunk.semantic_score),
                    bm25_score=max(existing_chunk.bm25_score, scored_chunk.bm25_score),
                    rrf_score=new_score,
                    source="hybrid",
                )
                fused[content_key] = (new_score, merged)
            else:
                scored_chunk_copy = ScoredChunk(
                    document=scored_chunk.document,
                    semantic_score=scored_chunk.semantic_score,
                    bm25_score=scored_chunk.bm25_score,
                    rrf_score=rrf_contribution,
                    source=scored_chunk.source,
                )
                fused[content_key] = (rrf_contribution, scored_chunk_copy)

    # Sort by fused RRF score, descending
    results = [chunk for _, chunk in fused.values()]
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
    Run both semantic search and BM25 keyword search, then fuse the
    results using Reciprocal Rank Fusion.

    This is the main retrieval function called by all downstream
    consumers (tools, generation chains, follow-up question handling).

    Args:
        query: the user's search string.
        vectorstore: the ChromaDB vector store.
        k: number of final results to return after fusion.
        section_filter: if provided, only chunks with this section_type
                        are searched by both retrievers.

    Returns:
        List of ScoredChunk objects sorted by fused RRF score (highest
        first), capped at k results. Each ScoredChunk carries both the
        semantic and BM25 scores so the UI can display them.
    """
    # 1. Semantic search via ChromaDB
    semantic_results = semantic_search(
        query, vectorstore, k=config.SEMANTIC_TOP_K, section_filter=section_filter,
    )

    # 2. BM25 keyword search over all chunks (with optional section filter)
    all_chunks = get_all_chunks(vectorstore, section_filter=section_filter)
    bm25_results = bm25_search(query, all_chunks, k=config.BM25_TOP_K)

    # 3. Fuse both ranked lists via RRF
    fused = _reciprocal_rank_fusion(
        [semantic_results, bm25_results],
        k=config.RRF_K,
    )

    logger.info(
        f"Hybrid search for '{query[:50]}...' — "
        f"semantic: {len(semantic_results)}, "
        f"bm25: {len(bm25_results)}, "
        f"fused: {len(fused)}, "
        f"returned: {min(k, len(fused))}"
    )

    return fused[:k]


# =============================================================================
# Limitations retrieval (smart fallback for gap analysis)
# =============================================================================

# Semantic query used to find limitation-like content in Discussion chunks
# when no explicit Limitations section exists for a paper. This query is
# designed to match the kind of language authors use when discussing
# constraints, caveats, and future directions — even without using the
# word "limitation" explicitly.
_LIMITATIONS_SEMANTIC_QUERY = (
    "study limitations, methodological constraints, caveats, "
    "shortcomings, future research directions, recommendations "
    "for future work, what remains to be investigated"
)


def retrieve_limitations(
    topic: str,
    vectorstore: Chroma,
    k: int = config.FINAL_TOP_K,
) -> list[ScoredChunk]:
    """
    Retrieve limitation and future-work content from the knowledge base.

    This function is designed for the gap analysis tool (Step d). It
    combines two retrieval strategies:

    1. Direct retrieval from `limitations_future` chunks — these are
       either from explicit Limitations sections or from the keyword-
       based fallback that extracted limitation sentences from Discussion
       during ingestion.

    2. Semantic search over `discussion` chunks using a limitations-focused
       query — this catches limitation-like content that the keyword
       fallback missed because the author used different phrasing (e.g.
       "our findings should be interpreted with caution" instead of
       "a limitation of this study").

    Both result sets are fused via RRF so the best limitation content
    surfaces regardless of whether it came from an explicit section or
    was buried in the Discussion.

    The user's topic is also incorporated into the semantic query so
    results are topically relevant, not just generically about limitations.

    Args:
        topic: the user's neuroscience topic (used to focus retrieval).
        vectorstore: the ChromaDB vector store.
        k: number of final results to return after fusion.

    Returns:
        List of ScoredChunk objects containing limitation and future-work
        content, sorted by fused RRF score.
    """
    # Strategy 1: hybrid search over explicit limitations_future chunks
    lim_chunks = hybrid_search(
        topic,
        vectorstore,
        k=config.SEMANTIC_TOP_K,
        section_filter="limitations_future",
    )

    # Strategy 2: semantic search over discussion chunks using a
    # limitations-focused query combined with the user's topic.
    # This catches limitation content that was not extracted during
    # ingestion because no explicit heading or keyword was present.
    combined_query = f"{topic} — {_LIMITATIONS_SEMANTIC_QUERY}"
    discussion_lim = semantic_search(
        combined_query,
        vectorstore,
        k=config.SEMANTIC_TOP_K,
        section_filter="discussion",
    )

    # Fuse both strategies via RRF
    fused = _reciprocal_rank_fusion(
        [lim_chunks, discussion_lim],
        k=config.RRF_K,
    )

    logger.info(
        f"Limitations retrieval for '{topic[:50]}...' — "
        f"from limitations_future: {len(lim_chunks)}, "
        f"from discussion (semantic): {len(discussion_lim)}, "
        f"fused: {len(fused)}, "
        f"returned: {min(k, len(fused))}"
    )

    return fused[:k]
