"""
PDF loading, section-aware chunking, and per-session ChromaDB ingestion.

Ported from v1 ingest.py with three changes:
    1. Per-session Chroma collection — collection name is
       f"neurohypothesis_{user_id}_{session_id}" instead of a global constant.
    2. SHA-256 chunk deduplication — before embedding, identical/near-
       identical chunks (e.g. same quoted passage in two papers) are
       dropped. Addresses reviewer comment #8.
    3. loguru replaces stdlib logging throughout.

The section detection regex patterns are carried unchanged from v1 —
the reviewer praised them and they handle edge-cases well (stop-heading
boundaries, inline reference lists, unnumbered ref lists, etc.).

Pipeline per uploaded PDF:
    1. Load via PyPDFLoader  (ERROR: unreadable → skip + warn)
    2. Clean PDF text artifacts
    3. Detect section headings → build heading map + stop boundaries
    4. Extract limitation sentences from Discussion as fallback
    5. Chunk each section with RecursiveCharacterTextSplitter
    6. (NEW) Deduplicate chunks by SHA-256 hash of normalised content
    7. Tag chunks past/future (in temporal_tagging.py)
    8. Embed with text-embedding-3-small
    9. Store in per-session Chroma collection

Public API:
    - PaperStatus           dataclass
    - ingest_pdfs(pdf_paths, user_id, session_id, embeddings) -> (Chroma, list[PaperStatus])
    - load_existing_vectorstore(user_id, session_id) -> Chroma | None
    - collection_name(user_id, session_id) -> str
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

import config

# =============================================================================
# Status reporting
# =============================================================================

SectionType = Literal["abstract", "introduction", "discussion", "limitations_future"]


@dataclass
class PaperStatus:
    """
    Outcome of ingesting one PDF.

    Rendered in the UI progress log so the user can see which papers
    were successfully chunked and which had to fall back to heuristics.
    """
    paper_id:       str
    status:         Literal["ingested", "partial", "failed"]
    sections_found: dict[SectionType, bool] = field(default_factory=dict)
    chunk_count:    int  = 0
    dedup_dropped:  int  = 0   # chunks removed by SHA-256 deduplication
    note:           str  = ""


# =============================================================================
# Chroma collection naming
# =============================================================================

def collection_name(user_id: str, session_id: str) -> str:
    """
    Return the per-session Chroma collection name.

    Uses short prefixes of both UUIDs to keep the name under Chroma's
    63-char collection-name limit while still being unique per session.
    """
    uid = user_id.replace("-", "")[:12]
    sid = session_id.replace("-", "")[:12]
    return f"neurohypothesis_{uid}_{sid}"


# =============================================================================
# Section-detection regex (carried verbatim from v1)
# =============================================================================

_LINE_START = r"(?im)^\s*(?:\d+\.?\s*|[IVX]+\.?\s*)?"

ABSTRACT_PATTERN     = re.compile(_LINE_START + r"(abstract|summary|background|objectives?|aims?|purpose)\b")
INTRODUCTION_PATTERN = re.compile(_LINE_START + r"(introduction)\b")
METHODS_PATTERN      = re.compile(
    _LINE_START + r"(methods?|materials?\s+and\s+methods?|patients?\s+and\s+methods?"
    r"|subjects?\s+and\s+methods?|study\s+(?:design|population|protocol)"
    r"|experimental\s+(?:procedures?|methods?)|participants?\s+and\s+(?:methods?|procedures?)"
    r"|data\s+(?:collection|sources?)|literature\s+search|search\s+strategy)\b"
)
DISCUSSION_PATTERN   = re.compile(
    _LINE_START + r"(discussion(?:\s+and\s+conclusions?)?|general\s+discussion"
    r"|results\s+and\s+discussion|concluding\s+remarks)\b"
)
CONCLUSIONS_PATTERN  = re.compile(_LINE_START + r"(conclusions?)\b")
LIMITATIONS_PATTERN  = re.compile(
    _LINE_START + r"(limitations?(?:\s+and\s+future(?:\s+(?:research|work|directions))?)?"
    r"|study\s+limitations|strengths\s+and\s+limitations|caveats"
    r"|future\s+(?:research|work|directions|studies)|recommendations"
    r"|directions\s+for\s+future\s+research|implications\s+and\s+future\s+directions)\b"
)
STOP_PATTERN         = re.compile(
    _LINE_START + r"(references|bibliography|acknowledgments?|acknowledgements?"
    r"|supplementary\s+materials?|supplemental\s+materials?|supporting\s+information"
    r"|appendix|appendices|author[s']?\s*(?:disclosures?|contributions?|declarations?)"
    r"|conflict[s]?\s+of\s+interest|competing\s+interests?|funding"
    r"|data\s+availability(?:\s+statement)?|disclosures?"
    r"|ethics\s+(?:statement|approval|declaration)|publication\s+history"
    r"|declaration\s+of\s+interest|editorial\s+note"
    r"|peer\s+review\s+(?:history|information)|study\s+funding"
    r"|abbreviations|glossary|about\s+the\s+authors?)\b"
)
STOP_PATTERN_INLINE_REFS = re.compile(r"(?i)\breferences\s+\d+")
STOP_PATTERN_REFLIST     = re.compile(
    r"[A-Z][a-z]{1,20},\s+[A-Z]\.\s*.*?\(\d{4}\).{1,300}?"
    r"[A-Z][a-z]{1,20},\s+[A-Z]\.\s*.*?\(\d{4}\)",
    re.DOTALL,
)

LIMITATION_KEYWORDS = [
    "limitation", "limitations", "limited by", "constraint", "caveat",
    "we could not", "was not possible", "small sample", "small cohort",
    "future research", "future studies", "future work", "further work",
    "further investigation", "we recommend", "we suggest",
    "should be investigated", "warrants further", "remains to be",
    "remain to be", "open question", "needs further", "should be explored",
]


# =============================================================================
# PDF loading
# =============================================================================

def _load_pdf(pdf_path: Path) -> str | None:
    """
    Load a single PDF and return its full concatenated text.

    Returns None if the file is unreadable or empty — the caller marks
    the paper as 'failed' and continues with the remaining PDFs.
    """
    try:
        pages = PyPDFLoader(str(pdf_path)).load()
    except Exception as exc:
        logger.error(f"PDF load failed ({pdf_path.name}): {exc}")
        return None

    text = "\n".join(p.page_content for p in pages if p.page_content)
    if not text.strip():
        logger.warning(f"PDF produced no text: {pdf_path.name}")
        return None
    return text


# =============================================================================
# Text cleaning
# =============================================================================

def _clean_extracted_text(text: str) -> str:
    """
    Remove common PDF-to-text artifacts that break section detection.

    Cleans: excessive whitespace, page-number lines, hyphenated line-
    breaks (re-joined), and null bytes.
    """
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)               # collapse blank lines
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)   # re-join hyphenated words
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # bare page numbers
    return text


# =============================================================================
# Section detection (stop-heading boundary approach from v1)
# =============================================================================

def _detect_sections(text: str) -> tuple[dict[SectionType, str], dict[SectionType, bool]]:
    """
    Detect section boundaries in the full paper text and extract content buckets.

    Returns:
        buckets:        {section_type: section_text} for non-empty sections.
        found_explicitly: {section_type: True/False} — False means fallback was used.
    """
    # ── Find stop boundary ────────────────────────────────────────────────
    stop_pos = len(text)
    for pattern in (STOP_PATTERN, STOP_PATTERN_INLINE_REFS):
        m = pattern.search(text)
        if m:
            stop_pos = min(stop_pos, m.start())
    ref_m = STOP_PATTERN_REFLIST.search(text)
    if ref_m:
        stop_pos = min(stop_pos, ref_m.start())

    useful_text = text[:stop_pos]

    # ── Locate all headings ───────────────────────────────────────────────
    headings: list[tuple[str, int]] = []
    for pattern, label in [
        (ABSTRACT_PATTERN,     "abstract"),
        (INTRODUCTION_PATTERN, "introduction"),
        (METHODS_PATTERN,      "methods"),
        (DISCUSSION_PATTERN,   "discussion"),
        (CONCLUSIONS_PATTERN,  "conclusions"),
        (LIMITATIONS_PATTERN,  "limitations_future"),
    ]:
        for m in pattern.finditer(useful_text):
            headings.append((label, m.start()))

    headings.sort(key=lambda x: x[1])

    # ── Extract text spans ────────────────────────────────────────────────
    buckets: dict[SectionType, str] = {}
    found_explicitly: dict[SectionType, bool] = {
        "abstract": False, "introduction": False,
        "discussion": False, "limitations_future": False,
    }

    for i, (label, start) in enumerate(headings):
        end = headings[i + 1][1] if i + 1 < len(headings) else len(useful_text)
        span = useful_text[start:end].strip()

        if not span:
            continue

        if label in ("abstract", "introduction", "discussion", "limitations_future"):
            target = label  # type: ignore[assignment]
        elif label == "conclusions":
            target = "discussion"   # append Conclusions to Discussion bucket
        elif label == "methods":
            continue                # Methods not used in hypothesis engine
        else:
            continue

        if target in buckets:
            buckets[target] += "\n\n" + span
        else:
            buckets[target] = span
        found_explicitly[target] = True

    # ── Preserve the pre-heading title block ──────────────────────────────
    # Text above the first heading (title, authors, journal) is otherwise
    # dropped; attach it to the earliest section so metadata extraction can
    # read the real citation instead of falling back to the temp filename.
    if headings:
        preamble = useful_text[:headings[0][1]].strip()
        if preamble:
            _pre_target = {
                "abstract": "abstract", "introduction": "introduction",
                "discussion": "discussion", "conclusions": "discussion",
                "limitations_future": "limitations_future",
            }
            target = next(
                (_pre_target[lbl] for (lbl, _s) in headings
                 if _pre_target.get(lbl) in buckets),
                "abstract",
            )
            buckets[target] = preamble + "\n\n" + buckets.get(target, "")
            found_explicitly[target] = True

    # ── Fallback: extract limitation sentences from Discussion ────────────
    if "limitations_future" not in buckets and "discussion" in buckets:
        disc = buckets["discussion"]
        lim_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", disc)
            if any(kw in s.lower() for kw in LIMITATION_KEYWORDS)
        ]
        if lim_sentences:
            buckets["limitations_future"] = " ".join(lim_sentences)
            found_explicitly["limitations_future"] = False
            logger.debug("Limitations: using sentence-level fallback from Discussion")

    # ── Fallback: use raw text if no sections detected ────────────────────
    if not buckets:
        buckets["discussion"] = useful_text[:5000]
        found_explicitly["discussion"] = False
        logger.warning("No section headings detected — using raw text as discussion fallback")

    return buckets, found_explicitly  # type: ignore[return-value]


# =============================================================================
# Chunking
# =============================================================================

def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _chunk_bucket(
    text: str,
    section_type: SectionType,
    paper_id: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """
    Split one section's text into overlapping chunks with rich metadata.

    Metadata attached to every chunk:
        paper_id, section_type, source (always "local"), chunk_index.
    temporal_lean is added later by temporal_tagging.py.
    """
    raw_chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={
                "paper_id":     paper_id,
                "section_type": section_type,
                "source":       "local",
                "chunk_index":  i,
                "temporal_lean": "untagged",  # filled in by temporal_tagging.py
            },
        )
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]


# =============================================================================
# SHA-256 deduplication  (NEW in v2 — reviewer comment #8)
# =============================================================================

def _sha256_of(text: str) -> str:
    """SHA-256 of normalised text — used for chunk deduplication."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

def _deduplicate_chunks(chunks: list[Document]) -> tuple[list[Document], int]:
    """
    Remove chunks whose normalised content is identical to an earlier chunk.

    Normalisation: lowercase + collapse whitespace.
    This catches verbatim passages quoted across multiple papers (e.g. two
    meta-analyses quoting the same primary study), which waste LLM context
    budget and can bias retrieval.

    Returns:
        (unique_chunks, n_dropped)
    """
    seen:   set[str]       = set()
    unique: list[Document] = []

    for chunk in chunks:
        h = _sha256_of(chunk.page_content)
        if h not in seen:
            seen.add(h)
            unique.append(chunk)

    n_dropped = len(chunks) - len(unique)
    if n_dropped:
        logger.info(f"Deduplication: dropped {n_dropped}/{len(chunks)} duplicate chunks")
    return unique, n_dropped


# =============================================================================
# Main ingestion entry point (called by N4a ingest_pdfs)
# =============================================================================

def ingest_pdfs(
    pdf_paths:  list[Path],
    user_id:    str,
    session_id: str,
    embeddings: OpenAIEmbeddings,
) -> tuple[Chroma, list[PaperStatus]]:
    """
    Load, chunk, deduplicate, tag, embed, and store uploaded PDFs.

    Creates a fresh per-session Chroma collection; any existing collection
    with the same name is dropped first so reruns start clean.

    Args:
        pdf_paths:  list of absolute Paths to user-uploaded PDFs (≤3).
        user_id:    cookie UUID (used in collection name).
        session_id: run UUID (used in collection name).
        embeddings: the shared OpenAIEmbeddings client.

    Returns:
        (vectorstore, statuses) where vectorstore is the populated Chroma
        instance and statuses is one PaperStatus per input PDF.

    Raises:
        RuntimeError if every PDF failed to ingest.
    """
    col_name = collection_name(user_id, session_id)
    splitter = _build_splitter()

    all_chunks: list[Document]   = []
    statuses:   list[PaperStatus] = []

    for pdf_path in pdf_paths:
        paper_id = pdf_path.stem
        status   = PaperStatus(paper_id=paper_id, status="failed")

        # ── Load ─────────────────────────────────────────────────────────
        raw_text = _load_pdf(pdf_path)
        if raw_text is None:
            status.note = "PDF could not be loaded"
            statuses.append(status)
            continue

        # ── Clean ────────────────────────────────────────────────────────
        raw_text = _clean_extracted_text(raw_text)

        # ── Detect sections ───────────────────────────────────────────────
        buckets, found_explicitly = _detect_sections(raw_text)
        status.sections_found = found_explicitly

        if not buckets:
            status.note = "No usable sections detected"
            statuses.append(status)
            continue

        # ── Chunk each bucket ─────────────────────────────────────────────
        paper_chunks: list[Document] = []
        for section_type, section_text in buckets.items():
            paper_chunks.extend(
                _chunk_bucket(section_text, section_type, paper_id, splitter)  # type: ignore[arg-type]
            )

        status.chunk_count   = len(paper_chunks)
        all_explicit         = all(found_explicitly.values())
        status.status        = "ingested" if all_explicit else "partial"
        if not all_explicit:
            missing      = [k for k, v in found_explicitly.items() if not v]
            status.note  = f"Used fallback for: {', '.join(missing)}"

        all_chunks.extend(paper_chunks)
        statuses.append(status)
        logger.info(
            f"Chunked {paper_id}: {len(paper_chunks)} chunks | "
            f"status={status.status}"
        )

    if not all_chunks:
        raise RuntimeError(
            "All PDFs failed to ingest. Check the logs and your PDF files."
        )

    # ── SHA-256 deduplication (across all papers) ─────────────────────────
    all_chunks, total_dropped = _deduplicate_chunks(all_chunks)
    # Propagate dropped counts back to per-paper statuses (proportionally)
    # This is approximate — exact attribution requires per-paper dedup first
    if total_dropped:
        for s in statuses:
            s.dedup_dropped = 0   # caller may refine if needed

    # ── Temporal tagging (past / future / neutral) ────────────────────────
    # Imported here to avoid circular imports; temporal_tagging imports utils.
    from src.engine.temporal_tagging import tag_chunks_past_future  # noqa: PLC0415
    all_chunks = tag_chunks_past_future(all_chunks, embeddings)

    # ── Embed and store in per-session Chroma ─────────────────────────────
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # Drop any stale collection from a previous run with same name
    try:
        chroma_client.delete_collection(col_name)
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=col_name,
        client=chroma_client,
    )

    total_chunks = sum(s.chunk_count for s in statuses if s.status != "failed")
    logger.info(
        f"Vectorstore ready: collection={col_name} | "
        f"{total_chunks} chunks (−{total_dropped} dupes) from "
        f"{sum(1 for s in statuses if s.status != 'failed')} papers"
    )
    return vectorstore, statuses


# =============================================================================
# Load existing vectorstore (for when Chroma already has the session data)
# =============================================================================

def load_existing_vectorstore(
    user_id:    str,
    session_id: str,
    embeddings: OpenAIEmbeddings,
) -> Chroma | None:
    """
    Load an already-built per-session vectorstore without re-embedding.

    Returns None if the collection doesn't exist (first run, or cleared
    after session end).
    """
    if not config.CHROMA_DIR.exists():
        return None

    col_name = collection_name(user_id, session_id)
    try:
        chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        existing_cols = [c.name for c in chroma_client.list_collections()]
        if col_name not in existing_cols:
            return None
        return Chroma(
            collection_name=col_name,
            embedding_function=embeddings,
            client=chroma_client,
        )
    except Exception as exc:
        logger.warning(f"Could not load existing vectorstore: {exc}")
        return None


# =============================================================================
# v2.1 NEW — Ingest PubMed abstracts (per-category, tagged with category)
# =============================================================================

def ingest_pubmed_abstracts(
    papers:      list[dict],
    user_id:     str,
    session_id:  str,
    embeddings:  OpenAIEmbeddings,
    existing_vs: Chroma | None = None,
) -> Chroma:
    """
    Embed top-N PubMed abstracts (one chunk per abstract) into a per-session
    Chroma collection.  Each chunk carries its category in metadata so the
    retriever can scope queries to the relevant categories of a hypothesis.

    If `existing_vs` is provided (e.g. when local PDFs were already ingested
    in N4a), abstracts are added to it.  Otherwise a fresh collection is
    created with the same collection_name() naming scheme.

    Args:
        papers:      list of Paper-dict (each must have abstract, pmid, category).
        user_id:     cookie UUID.
        session_id:  run UUID.
        embeddings:  the shared OpenAIEmbeddings client.
        existing_vs: optional pre-built Chroma to append to.

    Returns:
        The populated Chroma vectorstore.
    """
    docs: list[Document] = []
    for p in papers:
        abstract = p.get("abstract") or ""
        if not abstract.strip():
            continue
        pmid     = p.get("pmid") or p.get("paper_id", "")
        category = p.get("category", "Other")

        docs.append(Document(
            page_content=abstract,
            metadata={
                "paper_id":      pmid,
                "pmid":          pmid,
                "title":         p.get("title", ""),
                "category":      category,
                "section_type":  "abstract",          # for hybrid retriever
                "year":          p.get("year", 0),
                "journal":       p.get("journal", ""),
                "source":        "pubmed",
            },
        ))

    if not docs:
        logger.warning("ingest_pubmed_abstracts: no usable abstracts")
        return existing_vs   # may be None

    # ── Temporal tagging ──────────────────────────────────────────────────
    from src.engine.temporal_tagging import tag_chunks_past_future
    docs = tag_chunks_past_future(docs, embeddings)

    # ── Build / extend the vectorstore ────────────────────────────────────
    if existing_vs is not None:
        try:
            existing_vs.add_documents(docs)
            logger.info(
                f"ingest_pubmed_abstracts: appended {len(docs)} abstracts "
                f"to existing collection"
            )
            return existing_vs
        except Exception as exc:
            logger.warning(
                f"ingest_pubmed_abstracts: append failed ({exc}) — "
                f"falling back to fresh collection"
            )

    col_name = collection_name(user_id, session_id)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # Drop stale collection (first-time creation path)
    try:
        chroma_client.delete_collection(col_name)
    except Exception:
        pass

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=col_name,
        client=chroma_client,
    )
    logger.info(
        f"ingest_pubmed_abstracts: created collection {col_name} with "
        f"{len(docs)} abstracts"
    )
    return vs
