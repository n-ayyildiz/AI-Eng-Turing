"""
PDF loading, section-aware chunking, and ChromaDB ingestion.

This module is the "write side" of the RAG pipeline. It runs once when
the user clicks "Build knowledge base" in the status row.

Pipeline per paper:
    1. Load PDF via PyPDFLoader (ERROR HANDLING: unreadable files are caught)
    2. Join all pages into one text blob
    3. Locate ALL section headings (useful + stop) to build a heading map
    4. Use stop headings (References, Acknowledgments, etc.) as boundaries
       that cap the useful sections — NOT as trim-everything-after points
    5. Extract limitation/recommendation sentences from Discussion as fallback
    6. Chunk each section with RecursiveCharacterTextSplitter
    7. Attach rich metadata (paper_id, section_type, source, chunk_index)
    8. Embed with OpenAI text-embedding-3-small
    9. Store in local persistent ChromaDB

Public API:
    - build_vectorstore() -> (Chroma, list[PaperStatus])
    - load_existing_vectorstore() -> Chroma | None
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb

import config

logger = logging.getLogger(__name__)

# =============================================================================
# Status reporting
# =============================================================================

SectionType = Literal["abstract", "introduction", "discussion", "limitations_future"]


@dataclass
class PaperStatus:
    """Outcome of ingesting one paper. Rendered in the UI as a status card."""
    paper_id: str
    status: Literal["ingested", "partial", "failed"]
    sections_found: dict[SectionType, bool] = field(default_factory=dict)
    chunk_count: int = 0
    note: str = ""


# =============================================================================
# Regex patterns for section detection
# =============================================================================

# All patterns are case-insensitive and anchored at the start of a line.
# Optional numeric or roman-numeral prefixes are tolerated so headings like
# "4. Discussion" or "IV Discussion" still match.

_LINE_START = r"(?im)^\s*(?:\d+\.?\s*|[IVX]+\.?\s*)?"

ABSTRACT_PATTERN = re.compile(
    _LINE_START
    + r"(abstract|summary|background"
    + r"|objectives?|aims?|purpose)\b"
)

# Introduction is used both as a boundary marker (to cap the abstract span)
# AND as a section to extract. It contains study hypotheses, aims, and
# background from prior work — strongly past-flavoured content.
INTRODUCTION_PATTERN = re.compile(
    _LINE_START + r"(introduction)\b"
)

# Methods heading — marks where Introduction ends. Introduction text
# runs from the Introduction heading to the first Methods-type heading.
METHODS_PATTERN = re.compile(
    _LINE_START
    + r"(methods?"
    + r"|materials?\s+and\s+methods?"
    + r"|patients?\s+and\s+methods?"
    + r"|subjects?\s+and\s+methods?"
    + r"|study\s+(?:design|population|protocol)"
    + r"|experimental\s+(?:procedures?|methods?)"
    + r"|participants?\s+and\s+(?:methods?|procedures?)"
    + r"|data\s+(?:collection|sources?)"
    + r"|literature\s+search"
    + r"|search\s+strategy)\b"
)

DISCUSSION_PATTERN = re.compile(
    _LINE_START
    + r"(discussion(?:\s+and\s+conclusions?)?"
    + r"|general\s+discussion"
    + r"|results\s+and\s+discussion"
    + r"|concluding\s+remarks)\b"
)

# "Conclusions" is handled separately from Discussion so that a standalone
# "Conclusions" section sitting after Discussion is captured and appended
# to the discussion bucket rather than being lost.
CONCLUSIONS_PATTERN = re.compile(
    _LINE_START + r"(conclusions?)\b"
)

LIMITATIONS_PATTERN = re.compile(
    _LINE_START
    + r"(limitations?(?:\s+and\s+future(?:\s+(?:research|work|directions))?)?"
    + r"|study\s+limitations"
    + r"|strengths\s+and\s+limitations"
    + r"|caveats"
    + r"|future\s+(?:research|work|directions|studies)"
    + r"|recommendations"
    + r"|directions\s+for\s+future\s+research"
    + r"|implications\s+and\s+future\s+directions)\b"
)

# "Stop headings" mark the boundary where useful paper content ends.
# These are used as upper bounds for the last useful section — any content
# from a stop heading onward is excluded from all buckets. This prevents
# References, Acknowledgments, Supplementary Material, Author Disclosures,
# etc. from bleeding into the Discussion or Limitations sections.
STOP_PATTERN = re.compile(
    _LINE_START
    + r"(references"
    + r"|bibliography"
    + r"|acknowledgments?"
    + r"|acknowledgements?"
    + r"|supplementary\s+materials?"
    + r"|supplemental\s+materials?"
    + r"|supporting\s+information"
    + r"|appendix"
    + r"|appendices"
    + r"|author[s']?\s*(?:disclosures?|contributions?|declarations?)"
    + r"|conflict[s]?\s+of\s+interest"
    + r"|competing\s+interests?"
    + r"|funding"
    + r"|data\s+availability(?:\s+statement)?"
    + r"|disclosures?"
    + r"|ethics\s+(?:statement|approval|declaration)"
    + r"|publication\s+history"
    + r"|declaration\s+of\s+interest"
    + r"|editorial\s+note"
    + r"|peer\s+review\s+(?:history|information)"
    + r"|study\s+funding"
    + r"|abbreviations"
    + r"|glossary"
    + r"|about\s+the\s+authors?)\b"
)

# Catches "References 1." mid-paragraph — PDF-to-text sometimes loses the
# line break before the references list, so "References" does not appear at
# the start of a line. Pattern: the word "References" followed immediately
# by whitespace and a digit (e.g. "References 1. Smith et al.").
# Case-insensitive, no line-start anchor needed.
STOP_PATTERN_INLINE_REFS = re.compile(
    r"(?i)\breferences\s+\d+"
)

# Catches the start of an unnumbered reference list when the PDF has no
# "References" heading. Looks for 2+ consecutive reference entries in
# standard citation format: "Surname, I., ... (YEAR)." — works whether
# entries are separated by newlines OR just spaces (PDF-to-text often
# runs references together with double spaces instead of line breaks).
STOP_PATTERN_REFLIST = re.compile(
    r"[A-Z][a-z]{1,20},\s+[A-Z]\.\s*"   # "Ban, Y." or "Barnes, D. E."
    r".*?\(\d{4}\)"                       # anything then "(2009)"
    r".{1,300}?"                          # up to 300 chars (citation text)
    r"[A-Z][a-z]{1,20},\s+[A-Z]\.\s*"   # second entry: "Barnes, D."
    r".*?\(\d{4}\)",                      # "(2011)"
    re.DOTALL,                            # so . matches newlines too
)

# Sentence-level keyword fallback: when no standalone Limitations/Future
# section exists, sentences from the Discussion mentioning these phrases
# are extracted and bundled into the `limitations_future` bucket.
LIMITATION_KEYWORDS = [
    "limitation", "limitations", "limited by", "constraint", "caveat",
    "we could not", "was not possible", "small sample", "small cohort",
    "future research", "future studies", "future work", "further work",
    "further investigation", "we recommend", "we suggest",
    "should be investigated", "warrants further", "remains to be",
    "remain to be", "open question", "needs further", "should be explored",
]


# =============================================================================
# Loading (ERROR HANDLING happens here)
# =============================================================================

def _load_pdf(pdf_path: Path) -> str | None:
    """
    Load a single PDF and return its full concatenated text.

    ERROR HANDLING: None is returned if the PDF is unreadable or empty.
    The caller marks the paper as 'failed' and moves on.
    """
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
    except Exception as e:
        logger.error(f"Failed to load {pdf_path.name}: {e}")
        return None

    if not pages:
        logger.warning(f"{pdf_path.name} produced zero pages")
        return None

    # All pages are joined into one continuous text blob so section headings
    # can be located regardless of where they sit across page boundaries.
    full_text = "\n".join(page.page_content for page in pages)
    return full_text


# =============================================================================
# PDF text cleaning (removes extraction artifacts before section detection)
# =============================================================================

# Journal header/footer lines — repeating text from page margins that
# the PDF parser includes in the body. Examples:
#   "Frontiers in Aging Neuroscience | www.frontiersin.org 6 January 2020 | Volume 12 | Article 5"
#   "Neurology® 2023;100:e1234-e1245"
_JOURNAL_HEADER_RE = re.compile(
    r"(?im)^.*(?:"
    r"www\.\w+\.org"
    r"|\bvolume\s+\d+\s*\|\s*(?:issue|article)\s+\d+"
    r"|\bdoi\s*:\s*10\.\d+"
    r")\s*.*$"
)

# Figure and table captions embedded between paragraphs.
# Matches lines starting with "FIGURE N" or "TABLE N" (case-insensitive)
# followed by a pipe or colon, which is the standard caption format.
_FIGURE_TABLE_RE = re.compile(
    r"(?im)^(?:FIGURE|FIG\.?|TABLE)\s+\d+\s*[|:.].*$"
)


def _clean_extracted_text(text: str) -> str:
    """
    Remove PDF-to-text artifacts from the raw extracted text.

    Strips:
    - Journal header/footer lines (repeating on every page)
    - Figure and table captions (embedded between paragraphs)
    - Excessive blank lines left after removal

    Called after _load_pdf() and before _detect_sections() so that
    section detection and chunking work on clean content.
    """
    # Remove journal headers/footers
    text = _JOURNAL_HEADER_RE.sub("", text)

    # Remove figure/table captions
    text = _FIGURE_TABLE_RE.sub("", text)

    # Collapse excessive blank lines left after removal (3+ → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


# =============================================================================
# Heading map construction
# =============================================================================

def _build_heading_map(text: str) -> list[tuple[int, str]]:
    """
    Locate ALL recognisable headings in the text — both useful sections
    and stop headings — and return them sorted by position.

    Each entry is (character_position, heading_type) where heading_type
    is one of: 'abstract', 'introduction', 'discussion', 'conclusions',
    'limitations_future', or 'stop'.

    This map is the foundation for section boundary detection. By knowing
    where every heading sits, the correct end of each section can be
    determined by looking at the next heading in sequence — regardless
    of whether it is a useful heading or a stop heading.
    """
    headings: list[tuple[int, str]] = []

    # Useful headings
    m = ABSTRACT_PATTERN.search(text)
    if m:
        headings.append((m.start(), "abstract"))

    # Introduction — now extracted as a section (not just a boundary marker).
    # Its span runs from the introduction heading to the methods heading.
    m = INTRODUCTION_PATTERN.search(text)
    if m:
        headings.append((m.start(), "introduction"))

    # Methods heading — marks the end of the introduction span.
    # Not extracted as a bucket, used only as a boundary.
    m = METHODS_PATTERN.search(text)
    if m:
        headings.append((m.start(), "methods"))

    m = DISCUSSION_PATTERN.search(text)
    if m:
        headings.append((m.start(), "discussion"))

    m = CONCLUSIONS_PATTERN.search(text)
    if m:
        headings.append((m.start(), "conclusions"))

    m = LIMITATIONS_PATTERN.search(text)
    if m:
        headings.append((m.start(), "limitations_future"))

    # Stop headings — only the FIRST one is needed because everything
    # after any stop heading is irrelevant.
    m = STOP_PATTERN.search(text)
    if m:
        headings.append((m.start(), "stop"))

    # Inline numbered references: catches "References 1." mid-paragraph
    # where the line break was lost in PDF-to-text conversion.
    m2 = STOP_PATTERN_INLINE_REFS.search(text)
    if m2:
        headings.append((m2.start(), "stop"))

    # Unnumbered reference list: catches consecutive citation entries
    # ("Surname, I., ... (YEAR)") when there is no "References" heading.
    m3 = STOP_PATTERN_REFLIST.search(text)
    if m3:
        headings.append((m3.start(), "stop"))

    headings.sort()
    return headings


def _span_for(heading_type: str, heading_map: list[tuple[int, str]], text_length: int) -> tuple[int, int] | None:
    """
    Given a heading type, return the (start, end) character span for that
    section. The end is the position of the NEXT heading in the map (any
    type) or end of text. Returns None if the heading type is not in the map.
    """
    for i, (pos, htype) in enumerate(heading_map):
        if htype == heading_type:
            end = heading_map[i + 1][0] if i + 1 < len(heading_map) else text_length
            return (pos, end)
    return None


# =============================================================================
# Section detection (GRACEFUL DEGRADATION happens here)
# =============================================================================

def _extract_abstract_fallback(text: str) -> str:
    """
    GRACEFUL DEGRADATION: when no 'Abstract' heading is found, the first
    ~2000 characters of the ORIGINAL text are used. Abstracts are
    essentially always near the top of a paper, so this heuristic is
    reliable in practice even without an explicit label.
    """
    return text[:2000]


def _extract_limitation_sentences(discussion_text: str) -> str:
    """
    GRACEFUL DEGRADATION: when no standalone Limitations section exists,
    sentences from the Discussion mentioning limitation or future-work
    keywords are extracted and joined into a pseudo-section.

    The split on sentence boundaries is deliberately crude — it is fine
    for a fallback whose only job is to surface a handful of relevant
    sentences to the retriever.
    """
    sentences = re.split(r"(?<=[.!?])\s+", discussion_text)
    matching = [
        s for s in sentences
        if any(kw.lower() in s.lower() for kw in LIMITATION_KEYWORDS)
    ]
    return " ".join(matching)


# Signal words that indicate a study aim, hypothesis, or research question.
# Searched within the last 1500 chars of the introduction section.
_AIM_KEYWORDS = [
    "hypothesised", "hypothesized",
    "aimed to", "aim of this study", "aim was", "aims to",
    "present study",
    "sought to",
    "objective was", "objective of this study", "objectives were",
    "purpose of this study", "purpose was",
    "we investigated", "this study investigated",
    "we examined", "this study examined",
    "we tested",
    "to determine whether", "to determine if",
    "to evaluate whether", "to evaluate if",
    "to assess whether", "to assess if",
    "to explore whether",
    "we aimed", "study aimed",
    "this review", "this meta-analysis", "this systematic review",
]


def _extract_intro_aim(tail: str) -> str:
    """
    Extract sentences containing study aim / hypothesis signal words
    from the introduction tail (last 1500 chars).

    Returns matching sentences joined together. Falls back to the
    last 700 characters if no signal words are found.
    """
    import re as _re
    # Split on sentence boundaries
    sentences = _re.split(r"(?<=[.!?])\s+", tail.strip())
    matching = [
        s for s in sentences
        if any(kw.lower() in s.lower() for kw in _AIM_KEYWORDS)
    ]
    if matching:
        return " ".join(matching)
    # Fallback: last 700 characters
    return tail[-700:] if len(tail) > 700 else tail


def _detect_sections(text: str) -> tuple[dict[str, str], dict[SectionType, bool]]:
    """
    Build the four target buckets: abstract, introduction, discussion,
    limitations_future.

    Strategy:
        1. Build a heading map of ALL headings (useful + stop) sorted by
           position in the text.
        2. For each useful heading, its span runs from its position to the
           start of the NEXT heading in the map (any type).
        3. Introduction runs from the introduction heading to the methods
           heading. Contains study hypotheses, aims, and prior work —
           strongly past-flavoured content.
        4. If "Conclusions" appears as a standalone heading after "Discussion",
           its text is appended to the discussion bucket.
        5. Fallbacks apply when explicit headings are missing.

    Returns:
        buckets: {section_type: text content}
        found_explicitly: {section_type: True if found via explicit heading,
                           False if fallback logic was used}
    """
    heading_map = _build_heading_map(text)
    text_length = len(text)

    buckets: dict[str, str] = {}
    found_explicitly: dict[SectionType, bool] = {
        "abstract": False,
        "introduction": False,
        "discussion": False,
        "limitations_future": False,
    }

    # --- Abstract ---
    MAX_ABSTRACT_LENGTH = 3000
    span = _span_for("abstract", heading_map, text_length)
    if span:
        raw_abstract = text[span[0]:span[1]].strip()
        buckets["abstract"] = raw_abstract[:MAX_ABSTRACT_LENGTH]
        found_explicitly["abstract"] = True
    else:
        buckets["abstract"] = _extract_abstract_fallback(text)

    span = _span_for("introduction", heading_map, text_length)
    if span:
        raw_intro = text[span[0]:span[1]].strip()
        # Take only the last 1500 characters — the study aim, hypothesis,
        # and research question are always stated at the end of the
        # introduction, just before Methods.
        buckets["introduction"] = raw_intro[-1500:] if len(raw_intro) > 1500 else raw_intro
        found_explicitly["introduction"] = True

    # --- Discussion ---
    span = _span_for("discussion", heading_map, text_length)
    if span:
        discussion_text = text[span[0]:span[1]].strip()
        found_explicitly["discussion"] = True
    else:
        discussion_text = ""

    conclusions_span = _span_for("conclusions", heading_map, text_length)
    if conclusions_span:
        conclusions_text = text[conclusions_span[0]:conclusions_span[1]].strip()
        if discussion_text:
            discussion_text = discussion_text + "\n\n" + conclusions_text
        else:
            discussion_text = conclusions_text
            found_explicitly["discussion"] = True

    buckets["discussion"] = discussion_text

    # --- Limitations / Future ---
    MAX_LIMITATIONS_LENGTH = 5000
    span = _span_for("limitations_future", heading_map, text_length)
    if span:
        raw_lim = text[span[0]:span[1]].strip()
        buckets["limitations_future"] = raw_lim[:MAX_LIMITATIONS_LENGTH]
        found_explicitly["limitations_future"] = True
    elif discussion_text:
        buckets["limitations_future"] = _extract_limitation_sentences(discussion_text)
    else:
        buckets["limitations_future"] = ""

    # Empty buckets are dropped so they do not produce zero-content chunks.
    buckets = {k: v for k, v in buckets.items() if v.strip()}

    return buckets, found_explicitly


# =============================================================================
# Chunking + metadata
# =============================================================================

def _chunk_bucket(
    text: str,
    section_type: SectionType,
    paper_id: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """
    Split one bucket into chunks and attach metadata to each chunk.

    Metadata attached:
        paper_id     - filename stem, used for filtering and citations
        section_type - abstract | discussion | limitations_future
        source       - 'local' (PubMed chunks will be tagged 'pubmed')
        chunk_index  - position within this section
    """
    chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={
                "paper_id": paper_id,
                "section_type": section_type,
                "source": "local",
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(chunks)
    ]


# =============================================================================
# Past/Future tagging (gap analysis foundation — embedding-based)
# =============================================================================

def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _tag_chunks_past_future(
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
    metadata_by_paper: dict[str, dict] | None = None,
) -> None:
    """
    Tag each chunk as "past", "future", or "neutral" using a hybrid approach:

    1. Deterministic by section type:
       - introduction → always "past" (contains study hypotheses and prior work)
       - limitations_future → always "future" (contains gaps and recommendations)

    2. Cosine similarity to reference descriptions for abstract and discussion
       chunks. These sections mix past findings with future implications, so
       embedding-based disambiguation is needed.

    3. Metadata grounding (optional): if key_findings or hypothesis text from
       the paper's metadata is provided, abstract/discussion chunks from that
       paper that are more similar to the metadata than to either temporal
       reference are tagged "past" — grounding tagging in extracted findings.

    Modifies chunks in place — adds temporal_lean, past_similarity,
    future_similarity, and temporal_source to each chunk's metadata.
    """
    if not chunks:
        return

    # Embed temporal reference descriptions
    temporal_ref_texts = list(config.TEMPORAL_REFERENCES.values())
    temporal_ref_keys = list(config.TEMPORAL_REFERENCES.keys())
    temporal_ref_vecs = embeddings.embed_documents(temporal_ref_texts)
    temporal_ref_map = dict(zip(temporal_ref_keys, temporal_ref_vecs))

    # Embed metadata grounding vectors per paper (key_findings + hypothesis)
    metadata_vecs: dict[str, list[float]] = {}
    if metadata_by_paper:
        for pid, meta in metadata_by_paper.items():
            grounding_text = " ".join(filter(None, [
                meta.get("key_findings", ""),
                meta.get("hypothesis", ""),
            ])).strip()
            if grounding_text:
                vecs = embeddings.embed_documents([grounding_text])
                metadata_vecs[pid] = vecs[0]

    # Embed all chunks in one batched call
    chunk_texts = [c.page_content for c in chunks]
    chunk_vecs = embeddings.embed_documents(chunk_texts)

    past_count = 0
    future_count = 0
    neutral_count = 0

    for chunk, chunk_vec in zip(chunks, chunk_vecs):
        section = chunk.metadata.get("section_type", "")
        paper_id = chunk.metadata.get("paper_id", "")

        past_sim = _cosine_sim(chunk_vec, temporal_ref_map["past"])
        future_sim = _cosine_sim(chunk_vec, temporal_ref_map["future"])

        # Deterministic tagging by section type
        if section == "introduction":
            temporal_lean = "past"
            temporal_source = "section_type"
            past_count += 1

        elif section == "limitations_future":
            temporal_lean = "future"
            temporal_source = "section_type"
            future_count += 1

        else:
            # Cosine similarity based (abstract + discussion)
            temporal_source = "embedding"

            # Metadata grounding: if chunk is more similar to this paper's
            # key_findings/hypothesis than to either reference description,
            # tag as "past" — it's describing something already studied.
            if paper_id in metadata_vecs:
                meta_sim = _cosine_sim(chunk_vec, metadata_vecs[paper_id])
                if meta_sim > past_sim and meta_sim > future_sim:
                    temporal_lean = "past"
                    temporal_source = "metadata"
                    past_count += 1
                    chunk.metadata["temporal_lean"] = temporal_lean
                    chunk.metadata["past_similarity"] = round(past_sim, 3)
                    chunk.metadata["future_similarity"] = round(future_sim, 3)
                    chunk.metadata["temporal_source"] = temporal_source
                    continue

            diff = abs(past_sim - future_sim)
            if diff < config.TEMPORAL_NEUTRAL_MARGIN:
                temporal_lean = "neutral"
                neutral_count += 1
            elif past_sim > future_sim:
                temporal_lean = "past"
                past_count += 1
            else:
                temporal_lean = "future"
                future_count += 1

        chunk.metadata["temporal_lean"] = temporal_lean
        chunk.metadata["past_similarity"] = round(past_sim, 3)
        chunk.metadata["future_similarity"] = round(future_sim, 3)
        chunk.metadata["temporal_source"] = temporal_source

    logger.info(
        f"Past/Future tagging: {len(chunks)} chunks tagged — "
        f"{past_count} past, {future_count} future, {neutral_count} neutral"
    )


def _resolve_neutral_chunks_with_llm(chunks: list[Document]) -> None:
    """
    Run an LLM classification pass on chunks that the embedding tagged as
    "neutral" (past_sim and future_sim within TEMPORAL_NEUTRAL_MARGIN).

    The LLM decides whether each ambiguous chunk leans past or future.
    If the LLM also says "neutral", the tag stays neutral (genuinely mixed).

    Modifies chunks in place — updates `temporal_lean` and sets
    `temporal_source` to "llm" for chunks that were re-tagged.

    Cost: ~1 small LLM call per neutral chunk. For ~300 chunks with 20-30%
    neutral, that's 60-90 calls — roughly $0.01-0.02 with gpt-4o-mini.
    """
    # Lazy import to avoid circular dependency at module load time
    from src.generate import classify_neutral_chunk

    neutral_chunks = [c for c in chunks if c.metadata.get("temporal_lean") == "neutral"]
    if not neutral_chunks:
        logger.info("LLM fallback: no neutral chunks to classify")
        return

    logger.info(f"LLM fallback: classifying {len(neutral_chunks)} neutral chunks")

    # Mark embedding-origin tags on all non-neutral chunks
    for c in chunks:
        if c.metadata.get("temporal_lean") != "neutral":
            c.metadata.setdefault("temporal_source", "embedding")

    reclassified_past = 0
    reclassified_future = 0
    stayed_neutral = 0

    for chunk in neutral_chunks:
        llm_tag = classify_neutral_chunk(chunk.page_content)
        chunk.metadata["temporal_source"] = "llm"
        if llm_tag == "past":
            chunk.metadata["temporal_lean"] = "past"
            reclassified_past += 1
        elif llm_tag == "future":
            chunk.metadata["temporal_lean"] = "future"
            reclassified_future += 1
        else:
            stayed_neutral += 1

    logger.info(
        f"LLM fallback complete: "
        f"{reclassified_past} re-tagged as past, "
        f"{reclassified_future} re-tagged as future, "
        f"{stayed_neutral} stayed neutral"
    )


# =============================================================================
# Main entry points
# =============================================================================

def build_vectorstore() -> tuple[Chroma, list[PaperStatus]]:
    """
    Build the ChromaDB vector store from all PDFs in config.PAPERS_DIR.

    This is designed to be called once. If a vector store already exists
    on disk, load_existing_vectorstore() should be used instead.

    To force a clean rebuild: delete the chroma_db/ folder manually
    (rm -rf chroma_db) and restart the app.

    Returns:
        (vectorstore, [PaperStatus for each paper])
    """
    pdf_paths = sorted(config.PAPERS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {config.PAPERS_DIR}. "
            "Drop your papers in there and try again."
        )

    # A single splitter instance is reused across all papers — this is
    # stateless so it is safe and slightly more efficient than re-creating.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    statuses: list[PaperStatus] = []

    for pdf_path in pdf_paths:
        paper_id = pdf_path.stem
        status = PaperStatus(paper_id=paper_id, status="failed")

        # --- Step 1: load ---
        raw_text = _load_pdf(pdf_path)
        if raw_text is None:
            status.note = "PDF could not be loaded"
            statuses.append(status)
            continue

        # --- Step 1b: clean PDF artifacts ---
        raw_text = _clean_extracted_text(raw_text)

        # --- Step 2: detect sections (heading map + stop-heading boundaries) ---
        buckets, found_explicitly = _detect_sections(raw_text)
        status.sections_found = found_explicitly

        if not buckets:
            status.note = "No usable sections detected"
            statuses.append(status)
            continue

        # --- Step 3: chunk each bucket ---
        for section_type, section_text in buckets.items():
            chunks = _chunk_bucket(
                section_text,
                section_type,  # type: ignore[arg-type]
                paper_id,
                splitter,
            )
            all_chunks.extend(chunks)
            status.chunk_count += len(chunks)

        # --- Step 4: classify outcome ---
        # 'ingested' = all three sections located via explicit headings.
        # 'partial'  = at least one section was salvaged via fallback logic.
        all_explicit = all(found_explicitly.values())
        status.status = "ingested" if all_explicit else "partial"
        if not all_explicit:
            missing = [k for k, v in found_explicitly.items() if not v]
            status.note = f"Used fallback for: {', '.join(missing)}"

        statuses.append(status)

    if not all_chunks:
        raise RuntimeError(
            "All papers failed to ingest. Check the logs and your PDFs."
        )

    # --- Step 5: tag chunks as past/future (gap analysis foundation) ---
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    # Extract key_findings and hypothesis per paper at build time.
    # These are used as metadata grounding signals during tagging —
    # abstract/discussion chunks more similar to a paper's own findings
    # than to either temporal reference are tagged "past".
    # Cost: ~5 LLM calls (~$0.001 total) — runs once at build time.
    metadata_by_paper: dict[str, dict] = {}
    try:
        from src.generate import extract_paper_metadata
        papers_for_meta: dict[str, list] = {}
        for chunk in all_chunks:
            pid = chunk.metadata.get("paper_id", "unknown")
            if pid not in papers_for_meta:
                papers_for_meta[pid] = []
            papers_for_meta[pid].append(chunk)

        for pid, paper_chunks in papers_for_meta.items():
            chunks_text = "".join(
                f"[{c.metadata.get('section_type','?').upper()}] {c.page_content}\n\n"
                for c in paper_chunks
            )
            meta = extract_paper_metadata(pid, chunks_text)
            metadata_by_paper[pid] = meta
            logger.info(f"Build-time metadata extracted for {pid}")
    except Exception as e:
        logger.warning(f"Build-time metadata extraction failed: {e} — tagging without grounding")

    _tag_chunks_past_future(all_chunks, embeddings, metadata_by_paper=metadata_by_paper or None)

    # --- Step 6: embed and store ---
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
        client=chroma_client,
    )

    logger.info(
        f"Built vector store with {len(all_chunks)} chunks "
        f"from {len([s for s in statuses if s.status != 'failed'])} papers"
    )

    # Compute temporal tag counts from all_chunks for display in Ingestion tab
    past_count = sum(1 for c in all_chunks if c.metadata.get("temporal_lean") == "past")
    future_count = sum(1 for c in all_chunks if c.metadata.get("temporal_lean") == "future")
    neutral_count = sum(1 for c in all_chunks if c.metadata.get("temporal_lean") == "neutral")
    temporal_counts = {
        "past": past_count,
        "future": future_count,
        "neutral": neutral_count,
        "total": len(all_chunks),
    }

    # Persist ingestion statuses next to chroma_db/ so the Ingestion tab
    # shows the same info on subsequent app starts (not just right after
    # a fresh build). Saves chunk counts, section-found flags, and notes.
    _save_ingestion_statuses(statuses, temporal_counts)

    return vectorstore, statuses


def _statuses_json_path() -> Path:
    """Path to the persisted ingestion statuses JSON file."""
    return config.CHROMA_DIR.parent / f"{config.CHROMA_DIR.name}_statuses.json"


def _save_ingestion_statuses(statuses: list[PaperStatus], temporal_counts: dict | None = None) -> None:
    """Save ingestion statuses and temporal tag counts to JSON."""
    try:
        path = _statuses_json_path()
        data = {
            "temporal_counts": temporal_counts or {},
            "papers": [
                {
                    "paper_id": s.paper_id,
                    "status": s.status,
                    "chunk_count": s.chunk_count,
                    "sections_found": s.sections_found,
                    "note": s.note,
                }
                for s in statuses
            ],
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved ingestion statuses to {path}")
    except Exception as e:
        logger.warning(f"Could not save ingestion statuses: {e}")


def load_ingestion_statuses() -> list[PaperStatus]:
    """
    Load ingestion statuses from the JSON file saved by build_vectorstore.

    Returns an empty list if the file does not exist (first run before
    any build, or the file was deleted).
    """
    path = _statuses_json_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        # Handle both old format (list) and new format (dict with 'papers' key)
        papers = data.get("papers", data) if isinstance(data, dict) else data
        return [
            PaperStatus(
                paper_id=item.get("paper_id", "unknown"),
                status=item.get("status", "failed"),
                chunk_count=item.get("chunk_count", 0),
                sections_found=item.get("sections_found", {}),
                note=item.get("note", ""),
            )
            for item in papers
        ]
    except Exception as e:
        logger.warning(f"Could not load ingestion statuses: {e}")
        return []


def load_temporal_counts() -> dict:
    """
    Load temporal tag counts from the JSON file saved by build_vectorstore.

    Returns an empty dict if the file does not exist or has old format.
    """
    path = _statuses_json_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data.get("temporal_counts", {})
        return {}
    except Exception as e:
        logger.warning(f"Could not load temporal counts: {e}")
        return {}


def load_existing_vectorstore() -> Chroma | None:
    """
    Load an already-built vector store from disk without re-embedding.

    Returns None if no store exists yet — callers treat that as "the
    knowledge base has not been built yet, show the build button".
    """
    if not config.CHROMA_DIR.exists():
        return None

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        client=chroma_client,
    )
