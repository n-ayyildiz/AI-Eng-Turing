"""
LangGraph node functions for Neurohypothesis.
        n13_generate_hypothesis → passes previous_statements + hyp_index

Design contract (unchanged from v2):
    - Nodes never raise — all exceptions caught, logged, stored in state["errors"].
    - Vectorstore lives in module-level _VECTORSTORES dict (non-serialisable).
"""

from __future__ import annotations

import re
import time
from typing import Any

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.types import interrupt
from loguru import logger
from src.agent_state import AgentState

import config

# =============================================================================
# Module-level vectorstore cache
# =============================================================================

_VECTORSTORES: dict[str, Chroma] = {}


def _get_vs(session_id: str) -> Chroma | None:
    return _VECTORSTORES.get(session_id)


def _set_vs(session_id: str, vs: Chroma) -> None:
    _VECTORSTORES[session_id] = vs


def _embeddings() -> OpenAIEmbeddings:
    if not hasattr(_embeddings, "_client"):
        _embeddings._client = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    return _embeddings._client


def _timed(node_name: str, start: float) -> dict:
    return {"node_timings": {node_name: round(time.time() - start, 3)}}


def _sources_used_with_pdfs(state: AgentState, supported_by: list[str]) -> list[str]:
    """Citations for a hypothesis.

    The generator picks `supported_by` from the summaries alone (no chunk→paper
    provenance), so an uploaded PDF that genuinely fed the evidence is often
    missed.  In Path C (combined) we therefore union in the PDF paper_ids that
    were front-loaded into the summary window (N9), so the PDF appears in
    "Supporting evidence" — order preserved, deduped.  Other paths unchanged.
    """
    used = list(supported_by or [])
    if state.get("path_choice") == "combined":
        for pid in state.get("_path_c_pdf_pids", []) or []:
            if pid and pid not in used:
                used.append(pid)
    return used


# =============================================================================
# N1 — validate_input
# =============================================================================

def n1_validate_input(state: AgentState) -> dict:
    """Three-layer input guard: length + session throttle + OpenAI Moderation."""
    t = time.time()
    from src.tools.moderation import validate_input

    result = validate_input(
        topic=state.get("topic", ""),
        session_state=st.session_state,
    )
    update: dict[str, Any] = {"validation_passed": result.passed}
    if not result.passed:
        update["errors"] = [{
            "node":      "N1_validate_input",
            "type":      result.flagged_by,
            "message":   result.reason,
            "recovered": False,
        }]
        logger.warning(f"[N1] Validation failed: {result.flagged_by} — {result.reason}")
    update.update(_timed("N1", t))
    return update


# =============================================================================
# N2 — parse_topic
# =============================================================================

def n2_parse_topic(state: AgentState) -> dict:
    """LLM extracts {primary_method, primary_domain, focus} from raw topic."""
    t = time.time()
    from src.engine.generate import parse_topic

    parsed = parse_topic(state["topic"], node_name="N2_parse_topic")
    return {"parsed_topic": parsed.to_dict(), **_timed("N2", t)}


# =============================================================================
# N3 — route_path  (v2.1 replaces v2's route_sources)
# =============================================================================

def n3_route_sources(state: AgentState) -> dict:
    """
    Resolve the path choice from session state.

    The UI sets state["path_choice"] before the graph runs.  If the user
    uploaded PDFs but did not explicitly choose, we default to "combined".
    If no PDFs were uploaded, the choice is forced to "pubmed_only".
    """
    t = time.time()
    has_pdfs = bool(state.get("pdf_paths"))
    choice   = state.get("path_choice")

    if not choice:
        choice = "combined" if has_pdfs else "pubmed_only"
    if choice == "local_only" and not has_pdfs:
        logger.warning("[N3] path_choice=local_only but no PDFs — forcing pubmed_only")
        choice = "pubmed_only"

    legacy = {"local_only": "local_only",
              "pubmed_only": "pubmed_only",
              "combined":    "both"}[choice]

    logger.info(f"[N3] path_choice='{choice}' (has_pdfs={has_pdfs})")
    return {
        "path_choice":      choice,
        "source_decision":  legacy,
        **_timed("N3", t),
    }


# =============================================================================
# N4a — ingest_pdfs
# =============================================================================

def n4a_ingest_pdfs(state: AgentState) -> dict:
    """Load, chunk, deduplicate, tag, embed, store uploaded PDFs in Chroma."""
    t = time.time()
    from pathlib import Path

    from src.engine.chunking import ingest_pdfs

    pdf_paths = [Path(p) for p in state.get("pdf_paths", [])]
    if not pdf_paths:
        logger.warning("[N4a] No PDF paths in state — skipping ingest")
        return _timed("N4a", t)

    try:
        vs, statuses = ingest_pdfs(
            pdf_paths=pdf_paths,
            user_id=state["user_id"],
            session_id=state["session_id"],
            embeddings=_embeddings(),
        )
        _set_vs(state["session_id"], vs)
        logger.info(
            f"[N4a] Ingested {len(pdf_paths)} PDFs | "
            f"ok={sum(1 for s in statuses if s.status != 'failed')}"
        )
    except Exception as exc:
        logger.error(f"[N4a] ingest_pdfs failed: {exc}")
        return {
            "errors": [{"node": "N4a_ingest_pdfs", "type": "IngestError",
                        "message": str(exc), "recovered": False}],
            **_timed("N4a", t),
        }
    return _timed("N4a", t)


# =============================================================================
# N4b — retrieve_local  (diagnostic only)
# =============================================================================

def n4b_retrieve_local(state: AgentState) -> dict:
    """Run a test hybrid query to confirm the local vectorstore is populated."""
    t = time.time()
    from src.tools.retriever import hybrid_search

    vs = _get_vs(state["session_id"])
    if vs is None:
        logger.warning("[N4b] Vectorstore not found — skipping retrieve_local")
        return _timed("N4b", t)

    results = hybrid_search(state.get("topic", ""), vs, k=3)
    logger.info(f"[N4b] Local retrieval test: {len(results)} chunks returned")
    return _timed("N4b", t)


# =============================================================================
# N4c — extract_metadata_local  (kept; hidden from UI in v2.1)
# =============================================================================

def n4c_extract_metadata_local(state: AgentState) -> dict:
    """
    Extract structured metadata from each local PDF's chunks.

    In v2.1 the result drives gap analysis but is not exposed as a UI tab.
    """
    t = time.time()
    from langchain_core.prompts import ChatPromptTemplate
    from src.cost_tracking import count_tokens, get_tracker
    from src.tools.pubmed import _llm, _LocalPaperMetadataOutput
    from src.tools.retriever import get_all_chunks
    from src.utils import cosine_similarity, embed_texts

    vs = _get_vs(state["session_id"])
    if vs is None:
        return _timed("N4c", t)

    # Reference vectors for per-PDF categorization (cosine vs the 6 category
    # descriptions) and query relevance (cosine vs the user topic).  Embedded
    # once here, reused for every paper.  Used by Path C (combined) to rank and
    # gate PDF evidence; Path A ignores these fields.
    _cat_names = config.CATEGORIES
    try:
        _ref = embed_texts(
            [config.CATEGORY_DESCRIPTIONS[c] for c in _cat_names]
            + [state.get("topic", "")]
        )
        _cat_vecs  = _ref[:len(_cat_names)]
        _topic_vec = _ref[-1]
    except Exception as exc:
        logger.warning(f"[N4c] category/query reference embedding failed: {exc}")
        _cat_vecs, _topic_vec = None, None

    def _categorize(text: str) -> tuple[str | None, float]:
        """Return (best_category, query_cosine) for a PDF's representative text."""
        if not text or _cat_vecs is None or _topic_vec is None:
            return None, 0.0
        try:
            v = embed_texts([text])[0]
        except Exception:
            return None, 0.0
        sims = [cosine_similarity(v, cv) for cv in _cat_vecs]
        best = _cat_names[max(range(len(sims)), key=lambda i: sims[i])]
        return best, round(cosine_similarity(v, _topic_vec), 4)

    all_chunks = get_all_chunks(vs)
    papers_chunks: dict[str, list] = {}
    for chunk in all_chunks:
        pid = chunk.metadata.get("paper_id", "unknown")
        papers_chunks.setdefault(pid, []).append(chunk)
    # Order so the title block (abstract/intro) comes first — that's where the
    # citation lives — then discussion/limitations for findings & future work.
    _SEC_ORDER = {"abstract": 0, "introduction": 1, "discussion": 2, "limitations_future": 3}

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract structured metadata from the provided chunks of a scientific paper.\n"
         "First, read the title block on page 1 and extract VERBATIM (only if explicitly "
         "printed; never guess): the exact title, the author names in order, the journal/venue, "
         "and the publication year. Use 'Not available' (or an empty author list) when a field "
         "is not present in the chunks.\n"
         "Then summarise: topic, hypothesis, methods, key_findings, limitations, "
         "future_recommendations.\n"
         "'hypothesis' = the paper's main tested hypothesis or aim.\n"
         "'topic' is a short paraphrase, NOT the title.\n"
         "Each summary field: 1-2 sentences. Write 'Not available' if absent from chunks."),
        ("human", "Paper ID: {paper_id}\n\nChunks:\n{chunks_text}"),
    ])
    chain = prompt | _llm().with_structured_output(_LocalPaperMetadataOutput)

    metadata_list: list[dict] = []
    for pid, docs in papers_chunks.items():
        docs_sorted = sorted(
            docs,
            key=lambda d: (_SEC_ORDER.get(d.metadata.get("section_type", ""), 9),
                           d.metadata.get("chunk_index", 0)),
        )
        chunks_text = "\n\n".join(d.page_content for d in docs_sorted[:10])
        try:
            result: _LocalPaperMetadataOutput = chain.invoke({
                "paper_id":   pid,
                "chunks_text": chunks_text[:4000],
            })

            def _clean(v: str) -> str:
                v = (v or "").strip()
                return "" if v.lower() in ("", "not available", "n/a", "none") else v

            # Derive a clean filename fallback by stripping the random temp suffix.
            # e.g. "AGM2-6-71_dw5s57lo" → "AGM2-6-71" (the original upload name)
            name_parts = pid.rsplit("_", 1)
            clean_name = (name_parts[0]
                          if len(name_parts) > 1 and len(name_parts[-1]) <= 10
                          else pid)
            real_title = _clean(result.title)
            display_name = real_title or _clean(result.topic) or clean_name   # prefer real title, then topic paraphrase, then filename

            _cat, _qc = _categorize(
                " ".join(x for x in (display_name, result.topic, result.key_findings) if x)
            )

            meta = {
                "paper_id":   pid,
                "source":     "local",
                "title":      display_name,
                "authors":    [a for a in (result.authors or []) if a and a.strip()],
                "year":       _clean(result.year),
                "journal":    _clean(result.journal),
                "abstract":   result.key_findings,
                "mesh_terms": [],
                "publication_type": [],
                "category":     _cat,    # Path C: which scale/pairing (cosine vs descriptions)
                "query_cosine": _qc,     # Path C: relevance to the user query (rank + soft gate)
                "metadata": {
                    "topic":                  result.topic,
                    "hypothesis":             result.hypothesis,
                    "methods":                result.methods,
                    "key_findings":           result.key_findings,
                    "limitations":            result.limitations,
                    "future_recommendations": result.future_recommendations,
                },
            }
            metadata_list.append(meta)
            get_tracker().log_call(
                node_name="N4c_extract_metadata_local",
                model=config.MAIN_LLM_MODEL,
                call_type="llm",
                summary=f"extract local metadata: {pid}",
                input_tokens=count_tokens(chunks_text[:4000], config.MAIN_LLM_MODEL),
                output_tokens=count_tokens(result.topic, config.MAIN_LLM_MODEL),
            )
        except Exception as exc:
            logger.warning(f"[N4c] Metadata extraction failed for {pid}: {exc}")
            metadata_list.append({
                "paper_id": pid, "source": "local", "title": pid,
                "abstract": "", "mesh_terms": [], "publication_type": [],
                "category": None, "query_cosine": 0.0,
                "metadata": None,
            })

    logger.info(f"[N4c] Extracted metadata for {len(metadata_list)} local papers")
    return {"local_paper_metadata": metadata_list, **_timed("N4c", t)}


# =============================================================================
# N5_pick_primary  (v2.1 — replaces n5a-d primary query path)
# =============================================================================

def n5_pick_primary(state: AgentState) -> dict:
    """
    LLM picks the primary category by matching query meaning to category
    descriptions.  Deterministic (T=0, seeded).
    """
    t = time.time()
    from src.tools.pubmed import pick_primary_category

    category, rationale = pick_primary_category(
        topic=state["topic"],
        parsed_topic=state.get("parsed_topic", {}),
        node_name="N5_pick_primary",
    )
    logger.info(f"[N5_pick_primary] '{category}' | {rationale[:100]}")
    return {
        "primary_category": category,
        **_timed("N5_pick_primary", t),
    }


# =============================================================================
# N5_order_categories  (v2.1 — alphabetical complement ordering)
# =============================================================================

def n5_order_categories(state: AgentState) -> dict:
    """Build [primary, comp_a, comp_b, ...] with complements alphabetical."""
    t = time.time()
    from src.engine.generate import order_categories

    primary = state.get("primary_category") or "Behavioral & Cognitive Neuroscience"
    ordered = order_categories(primary)
    logger.info(f"[N5_order_categories] {ordered}")
    return {"ordered_categories": ordered, **_timed("N5_order_categories", t)}


# =============================================================================
# N5_per_category_retrieve  (v2.1 core — 3-attempt loop per category)
# =============================================================================

def n5_per_category_retrieve(state: AgentState) -> dict:
    """
    For each of the 6 ordered categories run the 3-attempt retrieval loop:

        reformulate → translate to MeSH → pre-search quality check →
        PubMed search → rank top-N by query cosine → per-abstract relevance.

    On success: store top-N papers in state.category_papers[category].
    On all 3 attempts failing but some passed: keep best-effort, badge as
    low-relevance.  If 0 passed across all attempts: skip the category.
    """
    t = time.time()
    from src.tools.pubmed import (
        RetrievalAttempt,
        check_retrieval_relevance,
        extract_metadata_pubmed,
        filter_predatory,
        quality_check_mesh,
        rank_top_n_by_query,
        reformulate_for_category,
        search_category,
        translate_to_mesh,
    )

    topic   = state["topic"]
    ordered = state.get("ordered_categories", [])
    if not ordered:
        logger.warning("[N5_per_cat] No ordered_categories — skipping retrieval")
        return _timed("N5_per_cat", t)

    category_papers:        dict[str, list] = {}
    category_relevance:     dict[str, dict] = {}
    category_reformulations: dict[str, list[dict]] = {}

    # ── Phase 1: Parallel retrieval (I/O-bound — no metadata LLM calls) ──
    # Each worker does: reformulate → MeSH → quality-check → PubMed search
    # → filter → rank → relevance check.  No extract_metadata_pubmed here —
    # that makes 10 LLM calls/category and would hit OpenAI rate limits when
    # run across 3 workers simultaneously.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _retrieve_one(category: str):
        """Retrieval-only worker (no metadata extraction)."""
        cat_attempts_log: list[dict] = []
        cat_kept_raw = None          # raw PubMedPaper list (no metadata yet)
        prev_ref = prev_cos = prev_nrel = None

        for attempt in range(config.REFORMULATE_MAX_ATTEMPTS):
            try:
                reformulation, _ = reformulate_for_category(
                    topic=topic, category=category, attempt=attempt,
                    prev_reformulation=prev_ref,
                    prev_mean_cosine=prev_cos,
                    prev_n_relevant=prev_nrel,
                    node_name=f"N5_reformulate.{category[:15]}",
                )
            except Exception as exc:
                logger.warning(f"[N5_per_cat] {category} reformulate failed (attempt {attempt}): {exc}")
                reformulation = topic   # fallback to original topic

            try:
                mesh_terms = translate_to_mesh(
                    reformulation=reformulation, category=category,
                    attempt=attempt, node_name=f"N5_mesh.{category[:15]}",
                )
            except Exception as exc:
                logger.warning(f"[N5_per_cat] {category} MeSH failed: {exc}")
                mesh_terms = []

            try:
                mesh_ok, mesh_score, _ = quality_check_mesh(
                    mesh_terms=mesh_terms, category=category,
                    node_name=f"N5_mesh_qc.{category[:15]}",
                )
            except Exception as exc:
                logger.warning(f"[N5_per_cat] {category} MeSH QC failed: {exc}")
                mesh_ok, mesh_score = True, 0.5   # allow through on error

            if not mesh_ok and attempt < config.REFORMULATE_MAX_ATTEMPTS - 1:
                cat_attempts_log.append(RetrievalAttempt(
                    attempt=attempt + 1,
                    temperature=config.REFORMULATE_TEMP_ESCALATION[attempt],
                    reformulation=reformulation, mesh_terms=mesh_terms,
                    quality_score=mesh_score, n_retrieved=0,
                    n_relevant=0, mean_cosine=0.0, passed=False,
                ).to_dict())
                prev_ref = reformulation
                prev_cos = 0.0
                prev_nrel = 0
                continue

            try:
                papers, _ = search_category(
                    reformulation=reformulation, mesh_terms=mesh_terms,
                    category=category, n=config.PUBMED_PER_CATEGORY_N,
                )
            except Exception as exc:
                logger.warning(f"[N5_per_cat] {category} PubMed search failed: {exc}")
                papers = []

            try:
                clean, _ = filter_predatory(papers)
                ranked   = rank_top_n_by_query(
                    papers=clean, query=topic, n=config.PUBMED_PER_CATEGORY_N
                )
                passed, n_pass, passing, mean_cos = check_retrieval_relevance(
                    papers=ranked,
                    threshold=config.RELEVANCE_THRESHOLD_PER_ABSTRACT,
                    min_pass=config.MIN_RELEVANT_ABSTRACTS,
                )
            except Exception as exc:
                logger.warning(f"[N5_per_cat] {category} ranking/relevance failed: {exc}")
                ranked = papers
                passed = False
                n_pass = 0
                passing = []
                mean_cos = 0.0

            cat_attempts_log.append(RetrievalAttempt(
                attempt=attempt + 1,
                temperature=config.REFORMULATE_TEMP_ESCALATION[attempt],
                reformulation=reformulation, mesh_terms=mesh_terms,
                quality_score=mesh_score, n_retrieved=len(ranked),
                n_relevant=n_pass, mean_cosine=mean_cos, passed=passed,
            ).to_dict())

            if passed:
                cat_kept_raw = passing
                break
            prev_ref = reformulation
            prev_cos = mean_cos
            prev_nrel = n_pass
            if attempt == config.REFORMULATE_MAX_ATTEMPTS - 1:
                cat_kept_raw = passing if passing else ranked
                if cat_kept_raw:
                    logger.warning(
                        f"[N5_per_cat] {category} exhausted retries — "
                        f"keeping {len(cat_kept_raw)} best-effort"
                    )

        return category, cat_kept_raw, cat_attempts_log

    # Run all 6 category retrievals in parallel.
    # NCBI HTTP calls are already serialised by _ncbi_rate_wait() regardless
    # of worker count, so 6 workers only adds LLM-side parallelism (safe).
    raw_results: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_retrieve_one, cat): cat for cat in ordered}
        for future in as_completed(futures):
            cat = futures[future]
            try:
                cat_out, cat_kept_raw, attempts = future.result()
                raw_results[cat_out] = (cat_kept_raw, attempts)
            except Exception as exc:
                logger.error(f"[N5_per_cat] {cat} worker crashed: {exc}")
                raw_results[cat] = (None, [])

    # ── Phase 2: Parallel metadata extraction (max 3 concurrent LLM workers)
    # Previously sequential; now runs up to 3 categories simultaneously.
    # Each worker makes ~10 sequential LLM calls internally → max 3 concurrent
    # OpenAI requests, well within gpt-4o-mini rate limits.
    def _extract_metadata_for(category: str) -> None:
        cat_kept_raw, attempts = raw_results.get(category, (None, []))
        category_reformulations[category] = attempts

        if cat_kept_raw:
            try:
                cat_kept = extract_metadata_pubmed(
                    cat_kept_raw, topic,
                    node_name=f"N5_metadata.{category[:15]}",
                )
            except Exception as exc:
                logger.warning(f"[N5_per_cat] {category} metadata failed: {exc}")
                cat_kept = cat_kept_raw

            mc = sum((p.query_cosine or 0.0) for p in cat_kept) / max(len(cat_kept), 1)
            category_papers[category]    = [p.to_state_dict() for p in cat_kept]
            last_attempt                 = attempts[-1] if attempts else {}
            category_relevance[category] = {
                "mean_cosine":         round(mc, 4),
                "n_relevant":          len(cat_kept),
                "n_retrieved":         last_attempt.get("n_retrieved", 0),
                "attempts":            len(attempts),
                "low_relevance_badge": not last_attempt.get("passed", False),
            }
            logger.info(
                f"[N5_per_cat] {category}: {len(cat_kept)} papers "
                f"(mean_cosine={mc:.3f})"
            )
        else:
            category_papers[category]    = []
            category_relevance[category] = {
                "mean_cosine": 0.0, "n_relevant": 0, "n_retrieved": 0,
                "attempts": len(attempts), "low_relevance_badge": True,
            }
            logger.warning(f"[N5_per_cat] {category}: no relevant papers found")

    with ThreadPoolExecutor(max_workers=3) as meta_executor:
        meta_futures = {
            meta_executor.submit(_extract_metadata_for, cat): cat
            for cat in ordered
        }
        for future in as_completed(meta_futures):
            cat = meta_futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error(f"[N5_per_cat] {cat} metadata worker crashed: {exc}")

    # ── Post-retrieval validation: stop if NO category returned any papers ─
    total_papers = sum(len(papers) for papers in category_papers.values())
    if total_papers == 0:
        logger.error(
            "[N5_per_cat] Zero papers retrieved across all 6 categories — "
            "stopping before hypothesis generation."
        )
        return {
            "category_papers":         category_papers,
            "category_relevance":      category_relevance,
            "category_reformulations": category_reformulations,
            "validation_passed": False,
            "errors": [{
                "node":      "N5_per_cat",
                "type":      "NoRelevantPapers",
                "message":   (
                    "No relevant papers were found in any of the 6 search categories. "
                    "Please try a different or broader research question."
                ),
                "recovered": False,
            }],
            **_timed("N5_per_cat", t),
        }

    return {
        "category_papers":         category_papers,
        "category_relevance":      category_relevance,
        "category_reformulations": category_reformulations,
        **_timed("N5_per_cat", t),
    }


# =============================================================================
# N5_embed_category_papers
# =============================================================================

def n5_embed_category_papers(state: AgentState) -> dict:
    """
    Push retrieved abstracts (tagged with their category) into the per-session
    Chroma vectorstore.  Coexists with any local PDFs already ingested in N4a.
    """
    t = time.time()
    from src.engine.chunking import ingest_pubmed_abstracts

    cat_papers = state.get("category_papers", {})
    flat_papers: list[dict] = []
    for cat, papers in cat_papers.items():
        for p in papers:
            p["category"] = cat
            flat_papers.append(p)

    if not flat_papers:
        logger.warning("[N5_embed] No PubMed abstracts to embed")
        return _timed("N5_embed", t)

    existing_vs = _get_vs(state["session_id"])
    try:
        vs = ingest_pubmed_abstracts(
            papers=flat_papers,
            user_id=state["user_id"],
            session_id=state["session_id"],
            embeddings=_embeddings(),
            existing_vs=existing_vs,
        )
        _set_vs(state["session_id"], vs)
        logger.info(f"[N5_embed] Ingested {len(flat_papers)} abstracts into Chroma")
    except Exception as exc:
        logger.error(f"[N5_embed] ingest_pubmed_abstracts failed: {exc}")
        return {
            "errors": [{"node": "N5_embed_category_papers",
                        "type": "EmbedError", "message": str(exc), "recovered": False}],
            **_timed("N5_embed", t),
        }
    return _timed("N5_embed", t)


# =============================================================================
# N8 — select_category_for_hypothesis  (v2.1: uses ordered_categories)
# =============================================================================

def n8_select_category_for_hypothesis(state: AgentState) -> dict:
    """
    Pick the primary + complement for the current hypothesis index.

    H1 (idx 0): primary only.
    H2..H6 (idx 1..5): primary + ordered_categories[idx].
    """
    t = time.time()
    from src.engine.generate import select_categories

    hyp_index = state.get("current_hypothesis_index", 0)
    ordered   = state.get("ordered_categories", [])
    primary, comp_cats = select_categories(ordered, hyp_index)

    # If the complement has zero papers, skip ahead so the hypothesis still
    # proceeds (with primary papers only) rather than crashing on empty evidence.
    # Always preserve the INTENDED complement so the UI can inform the user.
    cat_papers    = state.get("category_papers", {})
    intended_comp = list(comp_cats)          # save before potential drop

    # When ordered ran short (only 1 category had papers), select_categories
    # returns comp_cats=[] and intended_comp is also [].
    # Derive the intended complement name from the FULL category_papers dict
    # (all 6 categories are always present, even those with 0 papers).
    if not intended_comp and hyp_index > 0:
        all_sorted = sorted(
            cat_papers.keys(),
            key=lambda c: len(cat_papers.get(c, [])),
            reverse=True,
        )
        candidate = (
            all_sorted[hyp_index]
            if hyp_index < len(all_sorted) and all_sorted[hyp_index] != primary
            else next((c for c in all_sorted if c != primary), None)
        )
        if candidate:
            intended_comp = [candidate]

    if comp_cats and not cat_papers.get(comp_cats[0]):
        logger.warning(
            f"[N8] H{hyp_index+1}: complement '{comp_cats[0]}' has no papers "
            f"— dropping complement, generating from primary only"
        )
        comp_cats = []

    logger.info(
        f"[N8] H{hyp_index+1}: primary='{primary}' complementary={comp_cats}"
    )
    return {
        "current_hypothesis_index": hyp_index,
        "_current_primary_cat":           primary,
        "_current_comp_cats":             comp_cats,
        "_current_intended_comp_cats":    intended_comp,
        **_timed("N8", t),
    }


# =============================================================================
# N9 — retrieve_evidence
# =============================================================================

def n9_retrieve_evidence(state: AgentState) -> dict:
    """
    Retrieve past- and future-tagged evidence for the selected category pair.

    Three paths:
      Path B (PubMed):  reads LLM-extracted metadata from category_papers directly.
                        past  = paper.metadata.key_findings
                        future = paper.metadata.future_recommendations
                        Rationale: temporal tagging works on long PDF sections but
                        poorly on short PubMed abstracts (mixed tense in one paragraph).
      Path A (PDFs):    vectorstore retrieval with temporal-lean filter.
      Path C (combined): Path B metadata for PubMed papers, then supplements with
                         vectorstore chunks for PDFs (deduplicated by paper_id).
    """
    t = time.time()
    from src.tools.retriever import retrieve_by_temporal_tag

    vs       = _get_vs(state["session_id"])
    primary  = state.get("_current_primary_cat", "Behavioral & Cognitive Neuroscience")
    comp     = state.get("_current_comp_cats", [])
    cat_papers = state.get("category_papers", {})

    # ── Pick the right evidence source ─────────────────────────────────────
    use_metadata_path = bool(cat_papers and any(
        any(p.get("metadata") for p in cat_papers.get(c, []))
        for c in [primary] + comp
    ))

    past_chunks:   list[dict] = []
    future_chunks: list[dict] = []
    pdf_window_pids: list[str] = []   # Path C: PDF ids guaranteed in summary window

    if use_metadata_path:
        # Path B (and Path C PubMed branch): build evidence from per-paper metadata.
        relevant_papers: list[dict] = []
        for c in [primary] + comp:
            relevant_papers.extend(cat_papers.get(c, []))

        for p in relevant_papers:
            meta = p.get("metadata") or {}
            pid  = p.get("pmid") or p.get("paper_id", "")
            findings = (meta.get("key_findings") or "").strip()
            future   = (meta.get("future_recommendations") or "").strip()
            # past chunk: prefer key_findings, fall back to abstract
            if findings and findings.lower() != "not available":
                past_chunks.append({"text": findings, "paper_id": pid})
            elif p.get("abstract"):
                past_chunks.append({"text": p["abstract"], "paper_id": pid})
            # future chunk: only if explicit future recommendations present
            if future and future.lower() != "not available":
                future_chunks.append({"text": future, "paper_id": pid})

        logger.info(
            f"[N9] metadata path | H{state.get('current_hypothesis_index', 0)+1} "
            f"past={len(past_chunks)} future={len(future_chunks)} "
            f"papers={len(relevant_papers)}"
        )

        # ── Path C (combined): blend uploaded-PDF evidence with the PubMed
        # per-category evidence.  Each PDF carries (N4c):
        #   query_cosine — relevance to the user query  → primary rank + soft gate
        #   category     — which scale/pairing it belongs to → ordering boost
        # Policy (agreed): NEVER drop a PDF for category reasons; the only
        # exclusion is the soft query-relevance gate (clearly off-topic uploads,
        # surfaced as a UI warning).  Matching-category PDFs rank first; every
        # kept PDF is guaranteed at least one chunk (metadata signal is uncapped).
        # Only the extra RAG grounding chunks are capped so PDFs don't flood.
        if state.get("path_choice") == "combined":
            pair = [primary] + comp
            thr  = config.PATH_C_PDF_MIN_QUERY_COSINE
            cap  = config.PATH_C_PDF_MAX_CHUNKS
            seen = {(c["paper_id"], c["text"][:80])
                    for c in past_chunks + future_chunks if c.get("paper_id")}

            # PDF chunks are collected into their OWN lists, then front-loaded to
            # the head of past/future so they fall inside the SUMMARY_MAX_CHUNKS
            # window the summarizers + generator actually read (PubMed otherwise
            # fills that window first and the PDF never reaches the hypothesis).
            pdf_past: list[dict] = []
            pdf_future: list[dict] = []

            def _add(bucket: list[dict], text: str | None, pid: str) -> bool:
                txt = (text or "").strip()
                if not txt or txt.lower() == "not available":
                    return False
                key = (pid, txt[:80])
                if key in seen:
                    return False
                bucket.append({"text": txt, "paper_id": pid})
                seen.add(key)
                return True

            # Rank PDFs: query_cosine (primary) + category boost (ordering only).
            # Soft gate drops only PDFs below the query-relevance floor.
            ranked: list[tuple[float, dict]] = []
            n_gated = 0
            for p in state.get("local_paper_metadata", []):
                qc = float(p.get("query_cosine") or 0.0)
                if qc < thr:
                    n_gated += 1
                    continue
                boost = config.PATH_C_PDF_CATEGORY_BOOST if p.get("category") in pair else 0.0
                ranked.append((qc + boost, p))
            ranked.sort(key=lambda x: x[0], reverse=True)
            pdf_cat = {p.get("paper_id", ""): p.get("category") for _, p in ranked}

            # 1) Metadata signal — every kept PDF guaranteed ≥1 chunk (uncapped).
            for _, p in ranked:
                m   = p.get("metadata") or {}
                pid = p.get("paper_id", "")
                added = _add(pdf_past,   m.get("key_findings"),           pid)
                added = _add(pdf_past,   m.get("hypothesis"),             pid) or added
                added = _add(pdf_future, m.get("future_recommendations"), pid) or added
                added = _add(pdf_future, m.get("limitations"),            pid) or added
                if not added:                       # nothing distilled → fall back to abstract
                    _add(pdf_past, p.get("abstract"), pid)

            # 2) RAG grounding chunks — matching-category PDFs first, capped per tag.
            if vs is not None:
                for tag, bucket in (("past", pdf_past), ("future", pdf_future)):
                    scs = retrieve_by_temporal_tag(
                        tag, vs, k=config.FINAL_TOP_K, topic=state["topic"],
                    )
                    scs.sort(key=lambda sc: 1 if pdf_cat.get(sc.paper_id) in pair else 0,
                             reverse=True)
                    n_added = 0
                    for sc in scs:
                        if n_added >= cap:
                            break
                        if _add(bucket, sc.text, sc.paper_id):
                            n_added += 1

            # Front-load the top `cap` PDF chunks into the consumed window; any
            # extra PDF chunks go after PubMed (lower priority, may be truncated).
            past_chunks[:0]   = pdf_past[:cap]
            past_chunks.extend(pdf_past[cap:])
            future_chunks[:0] = pdf_future[:cap]
            future_chunks.extend(pdf_future[cap:])

            # PDF paper_ids guaranteed inside the summary window — credited in
            # sources_used at finalize so the PDF appears in "Supporting evidence"
            # even though the generator picks citations from summaries alone.
            pdf_window_pids = list({c["paper_id"]
                                    for c in pdf_past[:cap] + pdf_future[:cap]
                                    if c.get("paper_id")})

            logger.info(
                f"[N9] Path C PDF supplement | kept={len(ranked)} gated={n_gated} "
                f"front-loaded past={len(pdf_past[:cap])} future={len(pdf_future[:cap])} "
                f"(total past+={len(pdf_past)} future+={len(pdf_future)}; "
                f"query_cosine rank + category boost, gate<{thr})"
            )

    elif vs is not None:
        # ── Path A (PDF-only): combine LLM-distilled metadata with raw RAG chunks.
        # Metadata gives a clean denoised signal (robust on review papers where
        # temporal tagging is weak); RAG chunks add grounded verbatim detail.
        local_meta = state.get("local_paper_metadata", [])

        # 1) LLM-distilled metadata signal
        for p in local_meta:
            meta = p.get("metadata") or {}
            pid  = p.get("paper_id", "")
            findings = (meta.get("key_findings") or "").strip()
            tested   = (meta.get("hypothesis") or "").strip()
            future   = (meta.get("future_recommendations") or "").strip()
            limits   = (meta.get("limitations") or "").strip()
            for txt in (findings, tested):
                if txt and txt.lower() != "not available":
                    past_chunks.append({"text": txt, "paper_id": pid})
            if not findings and not tested and p.get("abstract"):
                past_chunks.append({"text": p["abstract"], "paper_id": pid})
            for txt in (future, limits):
                if txt and txt.lower() != "not available":
                    future_chunks.append({"text": txt, "paper_id": pid})

        # 2) Raw RAG chunks (grounding), deduped against the metadata signal
        seen = {(c["paper_id"], c["text"][:80]) for c in past_chunks + future_chunks}
        for tag, bucket in (("past", past_chunks), ("future", future_chunks)):
            for sc in retrieve_by_temporal_tag(
                tag, vs, k=config.FINAL_TOP_K, topic=state["topic"],
            ):
                key = (sc.paper_id, sc.text[:80])
                if key not in seen:
                    bucket.append({"text": sc.text, "paper_id": sc.paper_id})
                    seen.add(key)

        logger.info(
            f"[N9] Path A combined | past={len(past_chunks)} future={len(future_chunks)} "
            f"(metadata + RAG chunks, papers={len(local_meta)})"
        )

    else:
        logger.warning("[N9] No evidence source available — empty bundle")

    paper_ids = list({c["paper_id"] for c in past_chunks + future_chunks if c.get("paper_id")})
    return {
        "_evidence_past":   past_chunks,
        "_evidence_future": future_chunks,
        "_evidence_pids":   paper_ids,
        "_path_c_pdf_pids": pdf_window_pids,
        **_timed("N9", t),
    }


# =============================================================================
# N10 — summarize_past
# =============================================================================

def n10_summarize_past(state: AgentState) -> dict:
    """LLM summarises past-tagged evidence into 3 bullet points."""
    t = time.time()
    from src.engine.generate import EvidenceBundle, summarize_past

    evidence = EvidenceBundle(
        past_chunks=state.get("_evidence_past", []),
        future_chunks=state.get("_evidence_future", []),
        paper_ids=state.get("_evidence_pids", []),
        categories_used=[state.get("_current_primary_cat", "")]
                        + state.get("_current_comp_cats", []),
    )
    bullets = summarize_past(evidence, state["topic"], node_name="N10_summarize_past")
    return {"_past_summary": "\n".join(bullets), **_timed("N10", t)}


# =============================================================================
# N11 — summarize_future
# =============================================================================

def n11_summarize_future(state: AgentState) -> dict:
    """LLM summarises future-tagged evidence into 3 bullet points."""
    t = time.time()
    from src.engine.generate import EvidenceBundle, summarize_future

    evidence = EvidenceBundle(
        past_chunks=state.get("_evidence_past", []),
        future_chunks=state.get("_evidence_future", []),
        paper_ids=state.get("_evidence_pids", []),
        categories_used=[state.get("_current_primary_cat", "")]
                        + state.get("_current_comp_cats", []),
    )
    bullets = summarize_future(evidence, state["topic"], node_name="N11_summarize_future")
    return {"_future_summary": "\n".join(bullets), **_timed("N11", t)}


# =============================================================================
# N12 — compute_gap
# =============================================================================

def n12_compute_gap(state: AgentState) -> dict:
    """gap_score = 1 − cosine(past_summary, future_summary)."""
    t = time.time()
    from src.engine.generate import compute_gap

    gap = compute_gap(
        past_summary=state.get("_past_summary", ""),
        future_summary=state.get("_future_summary", ""),
    )
    return {"_gap_score": gap, **_timed("N12", t)}


# =============================================================================
# N13 — generate_hypothesis
# =============================================================================

def n13_generate_hypothesis(state: AgentState) -> dict:
    """LLM generates one hypothesis at T=0 with per-H seed and prior-context."""
    t = time.time()
    from src.engine.generate import generate_hypothesis

    hyp_index   = state.get("current_hypothesis_index", 0)
    gate_key    = str(hyp_index)
    attempts    = state.get("quality_gate_attempts", {}).get(gate_key, 0)
    fail_reason = state.get("_gate_failure_reason")

    previous = [
        h.get("text", "") for h in state.get("hypotheses", []) if h.get("text")
    ]

    output = generate_hypothesis(
        topic=state["topic"],
        primary_cat=state.get("_current_primary_cat", ""),
        comp_cats=state.get("_current_comp_cats", []),
        past_summary=state.get("_past_summary", ""),
        future_summary=state.get("_future_summary", ""),
        gap_score=state.get("_gap_score", 0.5),
        paper_ids=state.get("_evidence_pids", []),
        failure_reason=fail_reason,
        attempt=attempts,
        hyp_index=hyp_index,
        previous_statements=previous,
        node_name="N13_generate_hypothesis",
    )
    return {
        "_current_hypothesis": {
            "statement":          output.statement,
            "supported_by":       output.supported_by,
            "suggested_approach": output.suggested_approach,
        },
        **_timed("N13", t),
    }


# =============================================================================
# N14 — score_originality
# =============================================================================

def n14_score_originality(state: AgentState) -> dict:
    """Embedding-based originality: 1 − cosine(hypothesis, past_summary)."""
    t = time.time()
    from src.engine.generate import score_originality

    hyp    = state.get("_current_hypothesis", {})
    result = score_originality(
        hypothesis_text=hyp.get("statement", ""),
        past_summary=state.get("_past_summary", ""),
        embeddings=_embeddings(),
        node_name="N14_score_originality",
    )
    return {
        "_originality_result": {
            "score":  result.originality_score,
            "grade":  result.grade,
            "label":  result.grade_label,
            "passes": result.passes_gate,
        },
        **_timed("N14", t),
    }


# =============================================================================
# N15 — judge_plausibility
# =============================================================================

def n15_judge_plausibility(state: AgentState) -> dict:
    """LLM-as-judge: 6 dimensions, T=0, seeded."""
    t = time.time()
    from src.engine.generate import judge_plausibility

    hyp    = state.get("_current_hypothesis", {})
    result = judge_plausibility(
        hypothesis_text=hyp.get("statement", ""),
        paper_ids=hyp.get("supported_by", []),
        topic=state["topic"],
        node_name="N15_judge_plausibility",
    )
    return {
        "_plausibility_result": {
            "average":          result.average,
            "verdict":          result.verdict,
            "passes":           result.passes_gate,
            "scores":           result.scores,
            "improvement_tips": getattr(result, "improvement_tips", []),
        },
        **_timed("N15", t),
    }


# =============================================================================
# N16 — quality_gate  (Decision D)
# =============================================================================

def n16_quality_gate(state: AgentState) -> dict:
    """
    Deterministic gate: originality ≥ 0.2 AND plausibility ≥ 3.0.

    Decision D wiring:
        FAIL + attempts < 3      → quality_gate_passed=False → back to N13
        PASS, or attempts == 3   → quality_gate_passed=True  → onward to N17
    """
    t = time.time()
    from src.engine.generate import PlausibilityResult, quality_gate
    from src.engine.originality import _make_result

    hyp_index = state.get("current_hypothesis_index", 0)
    gate_key  = str(hyp_index)
    attempts  = state.get("quality_gate_attempts", {}).get(gate_key, 0)

    orig_data  = state.get("_originality_result", {})
    plaus_data = state.get("_plausibility_result", {})

    orig = _make_result(
        {"id": "H", "statement": state.get("_current_hypothesis", {}).get("statement", "")},
        similarity=1.0 - orig_data.get("score", 0.5),
    )
    plaus = PlausibilityResult(
        scores=plaus_data.get("scores", {}),
        average=plaus_data.get("average", 3.0),
        verdict=plaus_data.get("verdict", ""),
        passes_gate=plaus_data.get("passes", False),
    )

    decision = quality_gate(orig, plaus, attempt=attempts)

    new_attempts = {gate_key: attempts + 1}
    logger.info(
        f"[N16] H{hyp_index+1} attempt={attempts+1} "
        f"passes={decision.passes} best_of={decision.best_of_attempts}"
    )

    update: dict = {
        "quality_gate_passed":     decision.passes,
        "quality_gate_is_best_of": decision.best_of_attempts,
        "quality_gate_attempts":   new_attempts,
        "_gate_failure_reason":    decision.failure_reason if not decision.passes else None,
        **_timed("N16", t),
    }

    if decision.passes:
        hyp       = state.get("_current_hypothesis", {})
        orig_res  = state.get("_originality_result", {})
        plaus_res = state.get("_plausibility_result", {})

        primary  = state.get("_current_primary_cat", "")
        comp     = state.get("_current_comp_cats", [])
        intended_comp = state.get("_current_intended_comp_cats", comp)

        # PubMed-check evaluator (v1 freshness check) — runs once per accepted hyp
        pmc_result = _pubmed_check(
            hypothesis_text=hyp.get("statement", ""),
            topic=state["topic"],
        )

        # Build the full sources_retrieved list from category_papers
        cat_papers = state.get("category_papers", {})
        sources_retrieved: list[str] = []
        for c in [primary] + comp:
            for p in cat_papers.get(c, []):
                pid = p.get("pmid") or p.get("paper_id", "")
                if pid:
                    sources_retrieved.append(pid)

        # Contradictory evidence search (new — runs once per accepted hypothesis)
        from src.tools.pubmed import search_contradictory_evidence
        contra_result = search_contradictory_evidence(
            hypothesis_text=hyp.get("statement", ""),
            topic=state["topic"],
            node_name="N16_contradictory",
        )

        update["hypotheses"] = [{
            "index":                     hyp_index,
            "text":                      hyp.get("statement", ""),
            "statement":                 hyp.get("statement", ""),
            "primary_category":                    primary,
            "complementary_categories":            comp,
            "intended_complementary_categories":   intended_comp,
            "originality_score":         orig_res.get("score", 0.0),
            "originality_grade":         orig_res.get("label", "Moderately original"),
            "plausibility_avg":          plaus_res.get("average", 3.0),
            "plausibility_scores":       plaus_res.get("scores", {}),
            "plausibility_verdict":      plaus_res.get("verdict", ""),
            "improvement_tips":          plaus_res.get("improvement_tips", []),
            "pubmed_check_score":        pmc_result.get("score"),
            "pubmed_check_grade":        pmc_result.get("grade"),
            "pubmed_check_n_found":      pmc_result.get("n_found"),
            "pubmed_check_matches":      pmc_result.get("matches", []),
            "quality_gate_passes":       attempts + 1,
            "low_confidence":            decision.best_of_attempts,
            "sources_used":              _sources_used_with_pdfs(state, hyp.get("supported_by", [])),
            "sources_retrieved":         sources_retrieved,
            "past_summary":              state.get("_past_summary", ""),
            "future_summary":            state.get("_future_summary", ""),
            "gap_score":                 state.get("_gap_score", 0.0),
            "suggested_approach":        hyp.get("suggested_approach", []),
            "contradictory_evidence":    contra_result,
            "user_rating":               None,
            "user_comment":              None,
            "db_hyp_id":                 None,
        }]
    return update


_NOVELTY_QUERY_CACHE: dict[str, str] = {}
_NOVELTY_STOPWORDS = {
    "how", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an",
    "of", "in", "on", "to", "for", "and", "or", "with", "between", "by", "what",
    "which", "that", "this", "into", "from", "can", "will", "may", "affect",
    "affects", "affected", "effect", "effects", "impact", "impacts", "role",
    "relationship", "study", "its",
}


def _key_terms(topic: str) -> str:
    """Cheap fallback: strip filler/question words, keep content terms (AND-ed)."""
    words = re.findall(r"[A-Za-z\-]+", topic.lower())
    kept = [w for w in words if w not in _NOVELTY_STOPWORDS and len(w) > 2]
    return " ".join(kept) if kept else topic


def _novelty_search_query(topic: str) -> str:
    """
    Topic -> PubMed query for the novelty check, mirroring Path B: translate the
    topic to MeSH terms and OR-join them; fall back to content key-terms if MeSH
    is empty.  Cached per topic (the check is topic-level — the hypothesis enters
    only via the cosine rerank — so this runs once, not once per hypothesis).
    """
    if topic in _NOVELTY_QUERY_CACHE:
        return _NOVELTY_QUERY_CACHE[topic]
    query = _key_terms(topic)
    try:
        from src.tools.pubmed import translate_to_mesh
        mesh = translate_to_mesh(topic, category="general", node_name="N_novelty_mesh")
        if mesh:
            query = "(" + " OR ".join(f'"{m}"' for m in mesh) + ")"
    except Exception as exc:
        logger.warning(f"novelty MeSH build failed: {exc}")
    _NOVELTY_QUERY_CACHE[topic] = query
    logger.info(f"[novelty] query: {query[:120]}")
    return query


def _pubmed_check(
    hypothesis_text: str,
    topic:           str,
    years_back:      int   = config.PUBMED_CHECK_YEARS_BACK,
    pool_n:          int   = config.PUBMED_CHECK_TOP_N,
    threshold:       float = config.PUBMED_CHECK_MATCH_THRESHOLD,
    search_query:    str | None = None,
) -> dict:
    """
    External-novelty check (shared by Paths A, B, C).

    Search PubMed prior art for the topic, embed the abstracts, cosine-rank
    them against the hypothesis, and score by the NEAREST neighbour:
        originality = 1 - max_cosine
    Higher cosine to the closest paper => less original (already published).
    'matches' = papers with cosine >= threshold (shown to the user as
    similar existing work).
    """
    from src.tools.pubmed import _efetch, _esearch
    from src.utils import cosine_similarity, embed_texts, grade_similarity

    try:
        query = search_query or _novelty_search_query(topic)
        pmids = _esearch(query, n=pool_n, years_back=years_back)
        if not pmids:
            kt = _key_terms(topic)
            if kt and kt != query:
                logger.info(f"[novelty] MeSH query empty — retrying key-terms: {kt[:80]}")
                pmids = _esearch(kt, n=pool_n, years_back=years_back)
        if not pmids:
            return {"score": None, "grade": "no recent papers", "n_found": 0, "matches": []}
        papers = [p for p in _efetch(pmids[:pool_n]) if p.abstract]
        if not papers:
            return {"score": None, "grade": "no abstracts", "n_found": 0, "matches": []}

        vecs = embed_texts([hypothesis_text] + [p.abstract for p in papers])
        hyp_vec, paper_vecs = vecs[0], vecs[1:]
        scored = sorted(
            (
                {"title": p.title, "year": p.year, "url": p.url,
                 "similarity": round(cosine_similarity(hyp_vec, v), 4)}
                for p, v in zip(papers, paper_vecs)
            ),
            key=lambda d: d["similarity"],
            reverse=True,
        )
        max_sim = scored[0]["similarity"]
        grade   = grade_similarity(max_sim, context="originality")
        return {
            "score":          round(1.0 - max_sim, 4),     # nearest-neighbour novelty
            "grade":          grade["label"],
            "n_found":        len(papers),
            "max_similarity": max_sim,
            "matches":        [s for s in scored if s["similarity"] >= threshold],
        }
    except Exception as exc:
        logger.warning(f"PubMed-check eval failed: {exc}")
        return {"score": None, "grade": "eval failed", "n_found": 0, "matches": []}


# =============================================================================
# N_a_generate  (Path A — PDF-only batch generation + scoring)
# =============================================================================

def n_a_generate(state: AgentState) -> dict:
    """
    Path A: generate ONE hypothesis per call, anchored on the next gap pair.

    On the first call (index 0) the ordered gap-pair list is built once via
    band-filtering + tiered/MMR selection and stored in state.  Each call then
    takes its slot's anchor pair, generates a hypothesis (with previous
    hypotheses injected for distinctness), runs a diversity gate (regenerate
    once if too similar to a prior), scores it with the SAME functions as B/C,
    and appends it.  N17/N18 present + collect the rating, and Decision E loops
    back here on 'continue' until the user finishes or the cap is hit.
    """
    t = time.time()
    from src.engine.generate import generate_path_a_one, judge_plausibility, score_originality
    from src.engine.originality import (
        build_gap_pairs,
        max_similarity_to_existing,
        select_pairs_tiered_mmr,
    )
    from src.tools.pubmed import search_contradictory_evidence

    idx    = state.get("current_hypothesis_index", 0)
    topic  = state["topic"]
    past   = state.get("_past_summary", "")
    future = state.get("_future_summary", "")
    pids   = state.get("_evidence_pids", []) or [
        p.get("paper_id", "") for p in state.get("local_paper_metadata", [])
    ]
    emb = _embeddings()

    # ── Build the ordered gap-pair list once (first hypothesis only) ──────────
    update_pairs: dict = {}
    pairs_state = state.get("_gap_pairs_ordered")
    if pairs_state is None:
        # Pair the CLEAN 5-bullet summaries (one sentence each), not raw chunks,
        # so each anchor is readable and the per-card gap shows one bullet a side.
        past_bullets   = [b.strip("• ").strip() for b in past.split("\n")   if b.strip()]
        future_bullets = [b.strip("• ").strip() for b in future.split("\n") if b.strip()]
        past_items   = [{"text": b, "paper_id": ""} for b in past_bullets]
        future_items = [{"text": b, "paper_id": ""} for b in future_bullets]
        raw, past_vecs, future_vecs = build_gap_pairs(past_items, future_items)
        chosen = select_pairs_tiered_mmr(
            raw, past_vecs, future_vecs, n=config.PATH_A_MAX_HYPOTHESES,
        )
        pairs_state = [
            {"past_text": c.past_text, "future_text": c.future_text, "gap_score": c.gap_score}
            for c in chosen
        ]
        update_pairs["_gap_pairs_ordered"] = pairs_state
        logger.info(
            f"[N_a_generate] built {len(pairs_state)} gap-pairs from "
            f"{len(past_bullets)}×{len(future_bullets)} summary bullets"
        )

    # ── Pick this slot's anchor pair (fall back to shared summary if exhausted)
    if idx < len(pairs_state):
        anchor = pairs_state[idx]
        past_anchor, future_anchor, pair_gap = (
            anchor["past_text"], anchor["future_text"], anchor["gap_score"],
        )
    else:
        # Pairs exhausted: fall back to the first clean bullet each side.
        past_anchor   = next((b.strip("• ").strip() for b in past.split("\n")   if b.strip()), past)
        future_anchor = next((b.strip("• ").strip() for b in future.split("\n") if b.strip()), future)
        pair_gap = state.get("_gap_score", 0.5)

    previous = [h.get("text", "") for h in state.get("hypotheses", []) if h.get("text")]

    # ── Generate (+ one diversity-gate retry if too close to a prior) ─────────
    out = generate_path_a_one(
        topic, past_anchor, future_anchor, past, future, pair_gap, pids,
        previous_statements=previous, seed_offset=idx, node_name="N_a_generate",
    )
    if previous:
        sim = max_similarity_to_existing(out.statement, previous)
        if sim >= config.DIVERSITY_GATE_THRESHOLD:
            logger.info(f"[N_a_generate] H{idx+1} too similar (sim={sim:.2f}) — regenerating")
            out = generate_path_a_one(
                topic, past_anchor, future_anchor, past, future, pair_gap, pids,
                previous_statements=previous, seed_offset=idx + 100, node_name="N_a_generate",
            )

    # ── Score with the same functions as B/C ──────────────────────────────────
    orig   = score_originality(out.statement, past, emb, node_name="N_a_originality")
    plaus  = judge_plausibility(out.statement, out.supported_by, topic, node_name="N_a_plausibility")
    pmc    = _pubmed_check(out.statement, topic)
    contra = search_contradictory_evidence(out.statement, topic, node_name="N_a_contradictory")

    hyp = {
        "index":                     idx,
        "text":                      out.statement,
        "statement":                 out.statement,
        "primary_category":          "",
        "complementary_categories":  [],
        "originality_score":         orig.originality_score,
        "originality_grade":         orig.grade_label,
        "plausibility_avg":          plaus.average,
        "plausibility_scores":       plaus.scores,
        "plausibility_verdict":      plaus.verdict,
        "improvement_tips":          getattr(plaus, "improvement_tips", []),
        "pubmed_check_score":        pmc.get("score"),
        "pubmed_check_grade":        pmc.get("grade"),
        "pubmed_check_n_found":      pmc.get("n_found"),
        "pubmed_check_matches":      pmc.get("matches", []),
        "quality_gate_passes":       1,
        "low_confidence":            False,
        "sources_used":              out.supported_by,
        "sources_retrieved":         pids,
        # Per-card gap = this slot's anchor pair (one clean bullet each side).
        "past_summary":              past_anchor,
        "future_summary":            future_anchor,
        "gap_score":                 pair_gap,
        "suggested_approach":        out.suggested_approach,
        "contradictory_evidence":    contra,
        "user_rating":               None,
        "user_comment":              None,
        "db_hyp_id":                 None,
    }
    logger.info(f"[N_a_generate] H{idx+1} ready (gap={pair_gap:.2f})")
    return {"hypotheses": [hyp], **update_pairs, **_timed("N_a_generate", t)}


# =============================================================================
# N17 — present_to_user  (INTERRUPT — HITL)
# =============================================================================

def n17_present_to_user(state: AgentState) -> dict:
    """Pause and wait for user rating + continue/stop decision."""
    t = time.time()
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        logger.error("[N17] No hypothesis in state to present")
        return {"user_decision": "stop", **_timed("N17", t)}

    hyp_index   = state.get("current_hypothesis_index", 0)
    current_hyp = (hypotheses[hyp_index]
                   if hyp_index < len(hypotheses) else hypotheses[-1])
    is_last     = (hyp_index >= config.PATH_A_MAX_HYPOTHESES - 1
                   if state.get("path_choice") == "local_only"
                   else hyp_index >= config.MAX_HYPOTHESES - 1)

    feedback: dict = interrupt({
        "action":           "rate_and_decide",
        "hypothesis":       current_hyp,
        "hypothesis_index": hyp_index,
        "is_last":          is_last,
    })

    rating   = int(feedback.get("rating",   3))
    comment  = str(feedback.get("comment",  ""))
    decision = str(feedback.get("decision", "stop"))

    logger.info(f"[N17] H{hyp_index+1} rated {rating}/5 | decision='{decision}'")
    return {
        "user_decision": decision,
        "user_rating":   rating,
        "user_comment":  comment,
        **_timed("N17", t),
    }


# =============================================================================
# N18 — collect_feedback  (Decision E)
# =============================================================================

def n18_collect_feedback(state: AgentState) -> dict:
    """Persist rating to SQLite and increment hypothesis index."""
    t = time.time()
    from src.db import save_hypothesis, update_hypothesis_feedback

    hypotheses = state.get("hypotheses", [])
    rating     = state.get("user_rating",  3)
    comment    = state.get("user_comment", "")

    if hypotheses:
        last_hyp = hypotheses[-1]
        db_id    = last_hyp.get("db_hyp_id")

        if not db_id:
            try:
                db_id = save_hypothesis(state["session_id"], last_hyp)
                hypotheses[-1]["db_hyp_id"] = db_id
            except Exception as exc:
                logger.error(f"[N18] save_hypothesis failed: {exc}")

        if db_id and rating:
            try:
                update_hypothesis_feedback(db_id, rating, comment)
            except Exception as exc:
                logger.error(f"[N18] update_hypothesis_feedback failed: {exc}")

    next_index = state.get("current_hypothesis_index", 0) + 1
    logger.info(
        f"[N18] Feedback persisted | next_index={next_index} | "
        f"decision='{state.get('user_decision')}'"
    )
    return {
        "current_hypothesis_index": next_index,
        **_timed("N18", t),
    }


# =============================================================================
# N19 — export_results
# =============================================================================

def n19_export_results(state: AgentState) -> dict:
    """Generate a PDF report containing all hypotheses + sources + scores."""
    t = time.time()
    try:
        from src.engine.exports import export_session_to_pdf
        pdf_path = export_session_to_pdf(
            topic=state["topic"],
            hypotheses=state.get("hypotheses", []),
            session_id=state["session_id"],
            cat_papers=state.get("category_papers", {}),
        )
        logger.info(f"[N19] PDF exported → {pdf_path}")
    except Exception as exc:
        logger.error(f"[N19] PDF export failed: {exc}")
    return _timed("N19", t)


# =============================================================================
# N20 — persist_session
# =============================================================================

def n20_persist_session(state: AgentState) -> dict:
    """Write session-end metadata to SQLite."""
    t = time.time()
    from src.db import close_session

    hypotheses = state.get("hypotheses", [])
    try:
        close_session(
            session_id=state["session_id"],
            n_hypotheses=len(hypotheses),
            completed=True,
        )
        logger.info(f"[N20] Session closed | {len(hypotheses)} hypotheses generated")
    except Exception as exc:
        logger.error(f"[N20] close_session failed: {exc}")
    return _timed("N20", t)
