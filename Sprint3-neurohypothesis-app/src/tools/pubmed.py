"""
PubMed E-utilities tool for Neurohypothesis v2.1.

Covers the redesigned per-category retrieval pipeline:

    pick_primary_category   — LLM matches user query meaning to one of 6
                              category descriptions (deterministic, T=0+seed)
    reformulate_for_category — LLM rewrites the query for one specific
                              category's methods (temperature escalates
                              with retry attempt: 0.0 → 0.2 → 0.4)
    translate_to_mesh       — LLM converts a reformulation to MeSH-aligned
                              terms (hidden from UI)
    quality_check_mesh      — LLM scores how well MeSH terms cover the
                              category descriptor (pre-search gate)
    search_category         — esearch + efetch with 25-year window, falls
                              back to no-year filter if 0 papers found
    rank_top_n_by_query     — deterministic embedding-cosine ranking of
                              retrieved abstracts vs the original user query
    check_retrieval_relevance — per-abstract cosine ≥ threshold check;
                              returns the count passing the per-abstract
                              gate (decision: ≥ MIN_RELEVANT_ABSTRACTS → pass)
    filter_predatory        — denylist filter on journal/publisher name (kept)
    extract_metadata_pubmed — LLM extracts structured fields per abstract
                              (kept; internally used by gap analysis)

Design notes:
    - All LLM calls pass seed=config.LLM_SEED for reproducibility.
    - All HTTP calls use httpx with src.utils.with_retries.
    - All LLM calls use with_structured_output() + Pydantic.
    - Cost is logged to SessionCostTracker after every LLM call.
"""

from __future__ import annotations

# =============================================================================
# NCBI rate limiter (shared across all threads)
# =============================================================================
# NCBI enforces: 3 req/s without API key, 10 req/s with key.
# With parallel retrieval workers all firing _esearch + _efetch simultaneously,
# we exceed this and get 429 errors.  This lock + sleep enforces the limit.
import os as _os
import threading as _threading
import time as _time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field
from src.utils import cosine_similarity, embed_texts, with_retries

import config

_NCBI_LOCK      = _threading.Lock()
_NCBI_LAST_CALL = 0.0


def _ncbi_rate_wait() -> None:
    """Block until it is safe to make the next NCBI HTTP call."""
    global _NCBI_LAST_CALL
    # 10 req/s with key (0.12s gap), 3 req/s without (0.40s gap)
    # Use slightly more conservative gaps than the theoretical minimum.
    min_gap = 0.12 if _os.getenv("NCBI_API_KEY") else 0.40
    with _NCBI_LOCK:
        elapsed = _time.monotonic() - _NCBI_LAST_CALL
        if elapsed < min_gap:
            _time.sleep(min_gap - elapsed)
        _NCBI_LAST_CALL = _time.monotonic()


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class PubMedPaper:
    """One paper retrieved from PubMed E-utilities."""
    pmid:             str
    title:            str
    abstract:         str
    journal:          str
    year:             int
    authors:          list[str]              = field(default_factory=list)
    mesh_terms:       list[str]              = field(default_factory=list)
    publication_type: list[str]              = field(default_factory=list)
    url:              str                    = ""
    category:         str | None             = None    # v2.1: set by which search retrieved this paper
    query_cosine:     float | None           = None    # v2.1: per-paper relevance score
    metadata:         dict | None            = None

    def to_state_dict(self) -> dict:
        """Convert to the Paper TypedDict-compatible dict for AgentState."""
        return {
            "pmid":             self.pmid,
            "source":           "pubmed",
            "title":            self.title,
            "abstract":         self.abstract,
            "full_text":        None,
            "journal":          self.journal,
            "year":             self.year,
            "authors":          self.authors,
            "mesh_terms":       self.mesh_terms,
            "publication_type": self.publication_type,
            "url":              self.url,
            "category":         self.category,
            "query_cosine":     self.query_cosine,
            "metadata":         self.metadata,
        }


@dataclass
class RetrievalAttempt:
    """One attempt of the per-category retrieval loop (for state.category_reformulations)."""
    attempt:       int
    temperature:   float
    reformulation: str
    mesh_terms:    list[str]
    quality_score: float | None       # pre-search LLM quality score, 0-1
    n_retrieved:   int                # how many PubMed returned after ranking to top-N
    n_relevant:    int                # how many passed per-abstract cosine gate
    mean_cosine:   float              # mean cosine of top-N to user query
    passed:        bool               # whether this attempt's retrieval was usable

    def to_dict(self) -> dict:
        return {
            "attempt":       self.attempt,
            "temperature":   self.temperature,
            "reformulation": self.reformulation,
            "mesh_terms":    self.mesh_terms,
            "quality_score": self.quality_score,
            "n_retrieved":   self.n_retrieved,
            "n_relevant":    self.n_relevant,
            "mean_cosine":   self.mean_cosine,
            "passed":        self.passed,
        }


# =============================================================================
# Pydantic output schemas (for with_structured_output)
# =============================================================================

class _PrimaryCategoryOutput(BaseModel):
    """Output of pick_primary_category."""
    category:  str = Field(description="Exactly one category name from the provided list")
    rationale: str = Field(description="One sentence explaining why this category best matches the query")


class _ReformulationOutput(BaseModel):
    """Output of reformulate_for_category."""
    reformulation: str = Field(description="Query rewritten for this category's methods (5-15 words)")
    rationale:     str = Field(description="One sentence explaining the rewording")


class _MeshTermsOutput(BaseModel):
    """Output of translate_to_mesh."""
    mesh_terms: list[str] = Field(description="MeSH-aligned terms (3-8 terms total)")


class _MeshQualityOutput(BaseModel):
    """Output of quality_check_mesh."""
    score:    float = Field(ge=0.0, le=1.0, description="Alignment score 0-1 (1 = perfect category alignment)")
    rationale: str  = Field(description="One sentence explaining the score")


class _PaperMetadataOutput(BaseModel):
    """Output of extract_metadata_pubmed."""
    topic:                  str = Field(description="Main research topic")
    hypothesis:             str = Field(description="The paper's main tested hypothesis or aim ('Not available' if absent)")
    methods:                str = Field(description="Key methods and study design")
    key_findings:           str = Field(description="Main results and outcomes")
    limitations:            str = Field(description="Study limitations")
    future_recommendations: str = Field(description="Suggested future research directions")


class _LocalPaperMetadataOutput(BaseModel):
    """
    Metadata for an uploaded PDF.  Adds bibliographic fields read VERBATIM from
    the paper's first page (title block) so citations show the real reference
    instead of a temp filename.  Extract only what is explicitly printed.
    """
    title:                  str       = Field(description="The paper's exact printed title, verbatim. 'Not available' if not present.")
    authors:                list[str] = Field(default_factory=list, description="Author names exactly as printed, in order. Empty list if not present.")
    year:                   str       = Field(description="Publication year if explicitly printed (e.g. in the header/footer/DOI), else 'Not available'. Never guess.")
    journal:                str       = Field(description="Journal or venue name if printed, else 'Not available'.")
    topic:                  str       = Field(description="Main research topic (a short paraphrase, not the title)")
    hypothesis:             str       = Field(description="The paper's main tested hypothesis or aim ('Not available' if absent)")
    methods:                str       = Field(description="Key methods and study design")
    key_findings:           str       = Field(description="Main results and outcomes")
    limitations:            str       = Field(description="Study limitations")
    future_recommendations: str       = Field(description="Suggested future research directions")


# =============================================================================
# LLM factory (v2.1: seed is always passed)
# =============================================================================

def _llm(
    temperature: float = config.MAIN_LLM_TEMPERATURE,
    seed:        int   = config.LLM_SEED,
) -> ChatOpenAI:
    """
    Return a ChatOpenAI client with seeded + temperature-controlled output.

    Every LLM call in v2.1 goes through this factory so reproducibility
    is enforced by construction.  Pass a derived seed (e.g. LLM_SEED +
    hyp_index) when you want reproducible-but-distinct outputs.
    """
    return ChatOpenAI(
        model=config.MAIN_LLM_MODEL,
        temperature=temperature,
        seed=seed,
    )


def _log(node_name: str, model: str, call_type: str, summary: str,
         prompt_text: str, output_text: str) -> None:
    """Log one LLM call to the SessionCostTracker (best-effort)."""
    try:
        from src.cost_tracking import count_tokens, get_tracker
        get_tracker().log_call(
            node_name=node_name,
            model=model,
            call_type=call_type,
            summary=summary,
            input_tokens=count_tokens(prompt_text, model),
            output_tokens=count_tokens(output_text, model),
        )
    except Exception as exc:
        logger.warning(f"Cost tracking failed in {node_name}: {exc}")


# =============================================================================
# HTTP client
# =============================================================================

def _get_client() -> httpx.Client:
    return httpx.Client(timeout=config.PUBMED_TIMEOUT_S)


# =============================================================================
# E-utilities: esearch (PMIDs) + efetch (full records)
# =============================================================================

def _esearch(query: str, n: int, years_back: int | None) -> list[str]:
    """
    Call NCBI esearch and return a list of PMIDs.

    If `years_back` is None, no date filter is applied (v2.1 fallback path
    when the windowed search returns 0 results).

    Raises httpx.HTTPError on network failure (caller retries via with_retries).
    """
    current_year = datetime.now().year

    if years_back is not None:
        start_year = current_year - years_back
        term       = f"({query}) AND {start_year}:{current_year}[dp]"
    else:
        term       = query

    params = {
        "db":      "pubmed",
        "term":    term,
        "retmax":  n,
        "sort":    "relevance",
        "retmode": "json",
        "email":   config.PUBMED_EMAIL,
    }
    if _api_key := _os.getenv("NCBI_API_KEY", "").strip():
        params["api_key"] = _api_key
    _ncbi_rate_wait()
    with _get_client() as client:
        # POST avoids URL-length limits for long Boolean queries (e.g. method keywords)
        resp = client.post(f"{config.PUBMED_BASE_URL}/esearch.fcgi", data=params)
    if resp.status_code == 429:
        logger.warning("NCBI esearch 429 — backing off 3s and retrying once")
        _time.sleep(3)
        _ncbi_rate_wait()
        with _get_client() as client:
            resp = client.post(f"{config.PUBMED_BASE_URL}/esearch.fcgi", data=params)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def _efetch(pmids: list[str]) -> list[PubMedPaper]:
    """Call efetch for a list of PMIDs and parse the XML response."""
    if not pmids:
        return []

    params = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "rettype": "xml",
        "email":   config.PUBMED_EMAIL,
    }
    if _api_key := _os.getenv("NCBI_API_KEY", "").strip():
        params["api_key"] = _api_key
    try:
        _ncbi_rate_wait()
        with _get_client() as client:
            resp = client.get(f"{config.PUBMED_BASE_URL}/efetch.fcgi", params=params)
        if resp.status_code == 429:
            logger.warning("NCBI efetch 429 — backing off 3s and retrying once")
            _time.sleep(3)
            _ncbi_rate_wait()
            with _get_client() as client:
                resp = client.get(f"{config.PUBMED_BASE_URL}/efetch.fcgi", params=params)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:
        logger.error(f"efetch failed for {len(pmids)} PMIDs: {exc}")
        return []

    papers: list[PubMedPaper] = []
    for article in root.findall(".//PubmedArticle"):
        paper = _parse_article_xml(article)
        if paper:
            papers.append(paper)
    return papers


def _parse_article_xml(article: ET.Element) -> PubMedPaper | None:
    """Parse one <PubmedArticle> XML element into a PubMedPaper."""
    pmid_el = article.find(".//PMID")
    pmid    = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
    if not pmid:
        return None

    title_el = article.find(".//ArticleTitle")
    title    = _el_text(title_el)

    abstract_parts = article.findall(".//AbstractText")
    abstract = " ".join(
        (f"[{p.get('Label', '')}] " if p.get("Label") else "") + (p.text or "")
        for p in abstract_parts
    ).strip()

    journal_el = article.find(".//Journal/Title")
    journal    = _el_text(journal_el)

    year_el = article.find(".//PubDate/Year")
    year    = int(year_el.text) if year_el is not None and year_el.text else 0
    if not year:
        medline_el = article.find(".//MedlineDate")
        if medline_el is not None and medline_el.text:
            import re
            m = re.search(r"(\d{4})", medline_el.text)
            year = int(m.group(1)) if m else 0

    authors: list[str] = []
    for author in article.findall(".//Author"):
        last  = _el_text(author.find("LastName"))
        first = _el_text(author.find("ForeName"))
        if last:
            authors.append(f"{last} {first}".strip())

    mesh_terms = [
        _el_text(d)
        for d in article.findall(".//MeshHeading/DescriptorName")
        if _el_text(d)
    ]

    pub_types = [
        _el_text(pt)
        for pt in article.findall(".//PublicationType")
        if _el_text(pt)
    ]

    return PubMedPaper(
        pmid=pmid,
        title=title,
        abstract=abstract,
        journal=journal,
        year=year,
        authors=authors,
        mesh_terms=mesh_terms,
        publication_type=pub_types,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


def _el_text(el: ET.Element | None) -> str:
    return el.text.strip() if el is not None and el.text else ""


# =============================================================================
# Predatory publisher filter (kept from v2)
# =============================================================================

def filter_predatory(
    papers: list[PubMedPaper],
) -> tuple[list[PubMedPaper], list[PubMedPaper]]:
    """
    Split papers into (clean, flagged) based on the predatory denylist.

    v2.1: NO publication-type filtering — Reviews, Editorials, Letters
    all remain in scope.  Only journal-name match against
    config.PREDATORY_PUBLISHERS is applied, plus dropping records with
    no year or no journal (almost always means a missing-metadata preprint).
    """
    clean:   list[PubMedPaper] = []
    flagged: list[PubMedPaper] = []

    for paper in papers:
        journal_lower = paper.journal.lower()
        is_predatory  = any(pub in journal_lower for pub in config.PREDATORY_PUBLISHERS)
        is_missing    = not paper.year or not paper.journal

        if is_predatory or is_missing:
            reason = "predatory publisher" if is_predatory else "missing year/journal"
            logger.debug(f"Filtered [{reason}]: {paper.title[:60]}… ({paper.journal})")
            flagged.append(paper)
        else:
            clean.append(paper)

    if flagged:
        logger.info(
            f"Predatory filter: {len(papers)} in → {len(clean)} clean, "
            f"{len(flagged)} flagged"
        )
    return clean, flagged


# =============================================================================
# v2.1 NEW — Primary category picker
# =============================================================================

_PICK_PRIMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You match a neuroscience research query to ONE neuroscience method "
     "category.  Read the user query and pick the SINGLE category whose "
     "description best matches what the user is asking about — by methods, "
     "domain, and underlying biology.\n\n"
     "Important: choose by MEANING and METHODS the query implies.  Do NOT "
     "guess based on what is most popular or commonly studied.  The primary "
     "category is the one the user's question is directly about.\n\n"
     "Available categories (you MUST return exactly one of these names):\n"
     "{category_block}\n\n"
     "Output exactly one category name."),
    ("human",
     "User query: {topic}\n\n"
     "Parsed components:\n"
     "  primary_method: {primary_method}\n"
     "  primary_domain: {primary_domain}\n"
     "  focus:          {focus}"),
])


def pick_primary_category(
    topic:        str,
    parsed_topic: dict[str, str],
    node_name:    str = "N_pick_primary_category",
) -> tuple[str, str]:
    """
    LLM picks the single neuroscience category that best matches the query.

    Returns:
        (category_name, rationale)
    Falls back to "Behavioral & Cognitive Neuroscience" on LLM failure
    (safe default — broad humans-focused bucket).
    """
    category_block = "\n".join(
        f"  • {name}: {desc[:280]}…"
        for name, desc in config.CATEGORY_DESCRIPTIONS.items()
    )
    prompt_text = (
        f"Query: {topic}\nMethod: {parsed_topic.get('primary_method', '')} "
        f"Domain: {parsed_topic.get('primary_domain', '')} "
        f"Focus: {parsed_topic.get('focus', '')}"
    )
    chain = _PICK_PRIMARY_PROMPT | _llm().with_structured_output(_PrimaryCategoryOutput)

    try:
        result: _PrimaryCategoryOutput = chain.invoke({
            "category_block": category_block,
            "topic":          topic,
            "primary_method": parsed_topic.get("primary_method", ""),
            "primary_domain": parsed_topic.get("primary_domain", ""),
            "focus":          parsed_topic.get("focus", ""),
        })
        category = result.category.strip()
        if category not in config.CATEGORIES:
            logger.warning(
                f"[{node_name}] LLM returned unknown category '{category}' — "
                f"falling back to Behavioral & Cognitive Neuroscience"
            )
            category = "Behavioral & Cognitive Neuroscience"
        _log(node_name, config.MAIN_LLM_MODEL, "llm",
             f"pick primary: {topic[:40]}", prompt_text, category)
        logger.info(f"[{node_name}] primary='{category}' | {result.rationale[:120]}")
        return category, result.rationale
    except Exception as exc:
        logger.error(f"[{node_name}] pick_primary_category failed: {exc} — fallback")
        return "Behavioral & Cognitive Neuroscience", f"fallback ({exc})"


# =============================================================================
# v2.1 NEW — Per-category reformulation
# =============================================================================

_REFORMULATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite a neuroscience research query so it targets ONE specific "
     "method category's literature.  The rewritten query will be passed to "
     "PubMed.\n\n"
     "TARGET CATEGORY: {category}\n"
     "CATEGORY DESCRIPTION:\n{category_desc}\n\n"
     "Rules:\n"
     "- 5-15 words\n"
     "- Use terminology aligned with the target category (its methods, "
     "  populations, model organisms, measurement types)\n"
     "- CRITICAL: Preserve BOTH the user's biological target (disease, "
     "  brain region, molecule, behaviour) AND the outcome/readout mentioned "
     "  in the original query.  For example, if the query mentions 'grey "
     "  matter', the rewritten query must also include a brain-structure term "
     "  (grey matter, cortical volume, brain atrophy, neurodegeneration, etc.) "
     "  even when adapting to a genetics or computational category.\n"
     "- Only adapt the METHODS / APPROACH to match the category — keep the "
     "  biological focus intact.\n"
     "- COMPUTATIONAL EXCEPTION: For Computational & Theoretical, DROP "
     "  imaging modality names (MRI, fMRI, EEG, DTI, PET, ultra-high field) — "
     "  these belong to the Neuroimaging category. Translate to computational "
     "  equivalents instead. E.g. 'visual pathways via ultra-high field MRI' "
     "  → 'computational model visual pathway connectivity processing'; "
     "  'resting-state fMRI default mode' → 'network model resting state "
     "  dynamics simulation'.\n"
     "- Do NOT use Boolean operators (AND/OR)\n"
     "- Do NOT add year filters — these are added automatically\n"
     "- If the user query is already aligned with this category, return it "
     "  largely unchanged.\n\n"
     "{prev_hint_block}"),
    ("human",
     "User query: {topic}"),
])

_REFORMULATE_HINT_TEMPLATE = (
    "PREVIOUS ATTEMPT INFO:\n"
     "- attempt: {attempt}\n"
     "- previous reformulation: {prev_reformulation}\n"
     "- previous mean cosine of retrieved abstracts vs user query: {prev_cosine:.3f}\n"
     "- previous n relevant of {n_top}: {prev_relevant}\n"
     "Improvement required: broaden the query or change terminology so more "
     "abstracts directly match the user's question.  Avoid the prior wording.\n\n"
)


def reformulate_for_category(
    topic:              str,
    category:           str,
    attempt:            int                              = 0,
    prev_reformulation: str | None                       = None,
    prev_mean_cosine:   float | None                     = None,
    prev_n_relevant:    int | None                       = None,
    node_name:          str                              = "N_reformulate_for_category",
) -> tuple[str, str]:
    """
    LLM rewrites the user query for one category's methods.

    Temperature escalates with attempt index (0 → 0.2 → 0.4).  Seed
    is fixed across attempts (LLM_SEED) — so attempt 1 of category X
    always produces the same reformulation given the same inputs.

    Returns:
        (reformulation_text, rationale)
    """
    temp = config.REFORMULATE_TEMP_ESCALATION[
        min(attempt, len(config.REFORMULATE_TEMP_ESCALATION) - 1)
    ]
    cat_desc = config.CATEGORY_DESCRIPTIONS.get(category, "")
    hint_block = ""
    if attempt > 0 and prev_reformulation is not None:
        hint_block = _REFORMULATE_HINT_TEMPLATE.format(
            attempt=attempt,
            prev_reformulation=prev_reformulation,
            prev_cosine=prev_mean_cosine or 0.0,
            n_top=config.PUBMED_PER_CATEGORY_N,
            prev_relevant=prev_n_relevant or 0,
        )

    chain = _REFORMULATE_PROMPT | _llm(temperature=temp).with_structured_output(_ReformulationOutput)
    prompt_text = f"Category: {category}\nQuery: {topic}"

    try:
        result: _ReformulationOutput = chain.invoke({
            "category":         category,
            "category_desc":    cat_desc,
            "topic":            topic,
            "prev_hint_block":  hint_block,
        })
        _log(node_name, config.MAIN_LLM_MODEL, "llm",
             f"reformulate ({category[:20]}) attempt={attempt+1} T={temp}",
             prompt_text, result.reformulation)
        logger.info(
            f"[{node_name}] {category[:25]} attempt={attempt+1} T={temp:.1f} "
            f"→ '{result.reformulation}'"
        )
        return result.reformulation, result.rationale
    except Exception as exc:
        logger.error(f"[{node_name}] reformulate failed for {category}: {exc} — using topic")
        return topic, f"fallback ({exc})"


# =============================================================================
# v2.1 NEW — MeSH translation (hidden from UI)
# =============================================================================

_MESH_TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Convert a neuroscience query into 4-6 MeSH-aligned terms suitable for "
     "PubMed search.\n\n"
     "Composition rules:\n"
     "- At least HALF of the terms must be METHOD-anchored for the target "
     "  category — i.e. terms that PubMed indexers use to TAG papers in "
     "  this category specifically.  Examples by category:\n"
     "  * Animal Models → 'Mice', 'Rats', 'Rodentia', "
     "'Disease Models, Animal', 'Behavior, Animal' (in-vivo work in animals)\n"
     "  * Human Neuroimaging → 'Magnetic Resonance Imaging', "
     "'Diffusion Magnetic Resonance Imaging', 'Electroencephalography', "
     "'Positron-Emission Tomography'\n"
     "  * Computational & Theoretical → 'Computer Simulation', "
     "'Models, Neurological', 'Neural Networks, Computer', 'Machine Learning'\n"
     "  * Genetics & Molecular Biology → 'Genome-Wide Association Study', "
     "'Gene Expression Profiling', 'Mice, Knockout', 'CRISPR-Cas Systems'\n"
     "  * Behavioral & Cognitive Neuroscience → 'Cognition', 'Memory', "
     "'Neuropsychological Tests', 'Psychophysics', 'Cohort Studies'\n"
     "  * Postmortem & Ex-Vivo Histology → 'Autopsy', "
     "'Postmortem Changes', 'Immunohistochemistry', 'Neurofibrillary Tangles', "
     "'Patch-Clamp Techniques', 'In Vitro Techniques' (any species; ex-vivo tissue work)\n"
     "- The remaining terms may be DOMAIN terms (the disease, brain region, "
     "  molecule, biomarker — e.g. 'Cholesterol, LDL', 'White Matter', "
     "  'Alzheimer Disease', 'Hippocampus').\n"
     "- Do NOT pad with generic terms like 'Brain', 'Humans', 'Animals', "
     "  'Neuroimaging', 'Cognition' unless they are central to the question.\n"
     "- Use exact MeSH descriptor names where possible (case and "
     "  punctuation matter: 'Cholesterol, LDL' not 'LDL Cholesterol').\n"
     "Output 4-6 terms total — no commas in your list separators, the "
     "caller joins them."),
    ("human",
     "Target category: {category}\n"
     "Query reformulation: {reformulation}"),
])


def translate_to_mesh(
    reformulation: str,
    category:      str,
    attempt:       int = 0,
    node_name:     str = "N_translate_to_mesh",
) -> list[str]:
    """
    LLM translates a reformulation into MeSH-aligned terms.

    Result is NOT shown to the user (kept internal — too verbose for the
    progress stream).  Temperature follows the attempt's escalation.

    Returns:
        List of MeSH terms (3-8 entries on success; empty list on failure).
    """
    temp = config.REFORMULATE_TEMP_ESCALATION[
        min(attempt, len(config.REFORMULATE_TEMP_ESCALATION) - 1)
    ]
    chain = _MESH_TRANSLATE_PROMPT | _llm(temperature=temp).with_structured_output(_MeshTermsOutput)

    try:
        result: _MeshTermsOutput = chain.invoke({
            "category":      category,
            "reformulation": reformulation,
        })
        _log(node_name, config.MAIN_LLM_MODEL, "llm",
             f"mesh ({category[:20]}) attempt={attempt+1}",
             reformulation, " ".join(result.mesh_terms))
        logger.debug(f"[{node_name}] {category[:25]} → mesh={result.mesh_terms}")
        return result.mesh_terms
    except Exception as exc:
        logger.error(f"[{node_name}] translate_to_mesh failed for {category}: {exc}")
        return []


# =============================================================================
# v2.1 NEW — Pre-search MeSH quality check
# =============================================================================

_MESH_QUALITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Score how well a list of MeSH terms covers the methods and concepts "
     "expected in a target neuroscience category.\n\n"
     "Return a score 0.0-1.0 where:\n"
     "  1.0 = MeSH terms strongly anchor in this category's typical methods\n"
     "  0.5 = mixed; some terms align, some are off-category\n"
     "  0.0 = MeSH terms do not match this category at all\n\n"
     "TARGET CATEGORY: {category}\n"
     "CATEGORY DESCRIPTION:\n{category_desc}"),
    ("human",
     "MeSH terms: {mesh_list}"),
])

# Threshold below which the reformulation is considered "not aligned with
# the target category" — drives the pre-search retry.
_MESH_QUALITY_PASS = 0.4


def quality_check_mesh(
    mesh_terms: list[str],
    category:   str,
    node_name:  str = "N_quality_check_mesh",
) -> tuple[bool, float, str]:
    """
    LLM scores the MeSH terms vs the target category descriptor.

    Returns:
        (passed, score, rationale) where passed = score >= _MESH_QUALITY_PASS.
        On LLM failure, returns (True, 0.5, "...") — fail-open so retrieval
        still runs and the more meaningful post-retrieval cosine check decides.
    """
    if not mesh_terms:
        return False, 0.0, "no MeSH terms produced"

    cat_desc = config.CATEGORY_DESCRIPTIONS.get(category, "")
    chain    = _MESH_QUALITY_PROMPT | _llm().with_structured_output(_MeshQualityOutput)

    try:
        result: _MeshQualityOutput = chain.invoke({
            "category":      category,
            "category_desc": cat_desc,
            "mesh_list":     ", ".join(mesh_terms),
        })
        passed = result.score >= _MESH_QUALITY_PASS
        _log(node_name, config.MAIN_LLM_MODEL, "llm",
             f"mesh quality ({category[:20]})",
             " ".join(mesh_terms), f"{result.score:.2f}")
        logger.debug(
            f"[{node_name}] {category[:25]} score={result.score:.2f} "
            f"passed={passed} | {result.rationale[:80]}"
        )
        return passed, result.score, result.rationale
    except Exception as exc:
        logger.warning(f"[{node_name}] mesh quality check failed: {exc} — fail-open")
        return True, 0.5, f"quality check unavailable ({exc})"


# =============================================================================
# v2.1 NEW — PubMed search per category (25y → no-year fallback)
# =============================================================================

def build_pubmed_query(reformulation: str, mesh_terms: list[str]) -> str:
    """
    Build a Boolean PubMed query from the LLM reformulation + MeSH terms.

    v2.1 fix (May 11, Option A): PubMed's automatic term mapping struggles
    with conversational English phrases like "in animal models" or
    "computational modeling of X".  Even when sensible MeSH terms exist
    (e.g. "Cholesterol, LDL", "Rodentia", "White Matter"), running the
    reformulation text alone returned 0 papers for 5 of 6 categories.

    This builder constructs a query that PubMed parses cleanly:

        (reformulation) AND ("MeSH term 1" OR "MeSH term 2" OR ...)

    Multi-word and comma-containing MeSH terms are quoted.  The
    reformulation provides specificity; the OR-clause of MeSH terms
    provides recall.

    If no MeSH terms are provided, falls back to the reformulation alone.
    """
    if not mesh_terms:
        return reformulation
    # Quote any MeSH term that contains a space or comma so PubMed treats
    # it as a literal phrase rather than a field qualifier.
    quoted = [
        (f'"{t}"' if (" " in t or "," in t) else t)
        for t in mesh_terms
    ]
    mesh_clause = "(" + " OR ".join(quoted) + ")"
    return f"({reformulation}) AND {mesh_clause}"


def build_pubmed_mesh_only_query(
    mesh_terms: list[str],
    category:   str | None = None,
) -> str:
    """
    Build a MeSH-only fallback query.  Used as a recall-maximising last
    resort when the AND-combined query returns 0 papers.

    When `category` is provided, the MeSH terms are
    FILTERED to those that anchor in this specific category according to
    config.MESH_TO_CATEGORY.  This drops generic domain terms (like
    "Cholesterol, LDL" or "White Matter" or "Brain") and keeps only
    category-method anchors (like "Computer Simulation" for Computational
    or "Mice" / "Rats" for Animal Models).

    Why: the previous MeSH-only fallback would OR all the LLM's MeSH terms
    together, including generic domain ones, which caused tangential
    papers to surface (e.g. hepatocyte cholesterol studies under a
    Computational-Neuroscience-of-LDL query).  The filtered version forces
    PubMed to return papers actually tagged with that category's methods.

    If the filter leaves fewer than 2 terms, returns an empty string,
    signalling the caller to NOT run a MeSH-only fallback for this category.
    This is the correct behaviour: if the LLM produced no category-specific
    terms, there's no clean way to widen the search without abandoning the
    category's distinctness.

    If `category` is None, no filtering happens (old behaviour, for callers
    that want the broad fallback).
    """
    if not mesh_terms:
        return ""

    terms_to_use = mesh_terms
    if category is not None:
        category_specific = {
            term for term, cat in config.MESH_TO_CATEGORY.items() if cat == category
        }
        terms_to_use = [t for t in mesh_terms if t in category_specific]
        if len(terms_to_use) < 2:
            # Not enough category-anchoring terms: refuse to run MeSH-only fallback.
            return ""

    quoted = [
        (f'"{t}"' if (" " in t or "," in t) else t)
        for t in terms_to_use
    ]
    return "(" + " OR ".join(quoted) + ")"


def search_category(
    reformulation: str,
    mesh_terms:    list[str],
    category:      str,
    n:             int = config.PUBMED_PER_CATEGORY_N,
) -> tuple[list[PubMedPaper], bool]:
    """
    Search PubMed for papers in one category using a layered Boolean strategy.

    v2.2 tier redesign: topic is ALWAYS present in every tier.
    The topic context (reformulation) is never dropped — only the method
    constraint is progressively widened:

        1. (reformulation) AND (LLM MeSH terms)           last 25y  — precise
        2. (reformulation) AND (LLM MeSH terms)           no year   — wider window
        3. (reformulation) AND (category method keywords) last 25y  — broad method
        4. (reformulation) AND (category method keywords) no year   — broadest
        5. (reformulation) alone                          no year   — last resort

    This replaces the old tiers 3-4 (MeSH-only, topic dropped) which caused
    low cosine scores because retrieved papers matched the method but not
    the biological topic.

    Returns:
        (papers, used_fallback_window)
    """
    safe_search = with_retries(
        _esearch,
        max_attempts=config.PUBMED_MAX_RETRIES,
        exceptions=(httpx.HTTPError, httpx.TimeoutException, Exception),
    )

    # PubMed indexes American English; normalise common British variants so
    # "grey matter" → "gray matter", "colour" → "color", etc.
    _BR_TO_US = {
        "grey":    "gray",
        "Grey":    "Gray",
        "colour":  "color",
        "Colour":  "Color",
        "oedema":  "edema",
        "Oedema":  "Edema",
        "favour":  "favor",
        "behaviour": "behavior",
        "Behaviour": "Behavior",
        "defence":  "defense",
        "Defence":  "Defense",
    }
    for br, us in _BR_TO_US.items():
        reformulation = reformulation.replace(br, us)

    combined_query = build_pubmed_query(reformulation, mesh_terms)
    method_kw      = config.CATEGORY_METHOD_KEYWORDS.get(category, "")
    method_query   = f'({reformulation}) AND ({method_kw})' if method_kw else ""

    pmids:          list[str] = []
    used_fallback:  bool      = False
    used_query:     str       = combined_query
    tier_succeeded: str       = ""

    # ── Tier 1: topic + LLM MeSH, 25-year window ─────────────────────────
    try:
        pmids = safe_search(combined_query, n, config.PUBMED_YEARS_BACK)
        if pmids:
            tier_succeeded = "combined-25y"
    except Exception as exc:
        logger.error(f"search_category({category}) tier-1 failed: {exc}")

    # ── Tier 2: topic + LLM MeSH, no year filter ─────────────────────────
    if not pmids and config.PUBMED_FALLBACK_NO_YEAR:
        try:
            pmids = safe_search(combined_query, n, None)
            if pmids:
                used_fallback = True
                tier_succeeded = "combined-noyear"
        except Exception as exc:
            logger.error(f"search_category({category}) tier-2 failed: {exc}")

    # ── Tier 3: topic + broad category method keyword, 25-year window ─────
    # Topic is preserved; method constraint is widened to simple keywords.
    if not pmids and method_query:
        try:
            pmids = safe_search(method_query, n, config.PUBMED_YEARS_BACK)
            if pmids:
                used_query = method_query
                tier_succeeded = "method-kw-25y"
        except Exception as exc:
            logger.error(f"search_category({category}) tier-3 failed: {exc}")

    # ── Tier 4: topic + broad category method keyword, no year filter ─────
    if not pmids and method_query and config.PUBMED_FALLBACK_NO_YEAR:
        try:
            pmids = safe_search(method_query, n, None)
            if pmids:
                used_query = method_query
                used_fallback = True
                tier_succeeded = "method-kw-noyear"
        except Exception as exc:
            logger.error(f"search_category({category}) tier-4 failed: {exc}")

    # ── Tier 5: plain reformulation, no year filter (last resort) ─────────
    if not pmids:
        try:
            pmids = safe_search(reformulation, n, None)
            if pmids:
                used_query = reformulation
                used_fallback = True
                tier_succeeded = "text-only-noyear"
        except Exception as exc:
            logger.error(f"search_category({category}) tier-5 failed: {exc}")

    if not pmids:
        logger.warning(
            f"search_category({category}): 0 PMIDs after all 5 tiers — "
            f"reformulation='{reformulation[:60]}'"
        )
        return [], used_fallback

    papers = _efetch(pmids)
    for p in papers:
        p.category = category

    logger.info(
        f"search_category({category}): tier='{tier_succeeded}' "
        f"query='{used_query[:80]}…' → {len(pmids)} PMIDs → {len(papers)} papers"
    )
    return papers, used_fallback


# =============================================================================
# v2.1 NEW — Rank top N by query similarity (deterministic, embedding cosine)
# =============================================================================

def rank_top_n_by_query(
    papers: list[PubMedPaper],
    query:  str,
    n:      int = config.PUBMED_PER_CATEGORY_N,
) -> list[PubMedPaper]:
    """
    Re-rank papers by cosine similarity of (title + abstract) to the user query.

    Pure embedding math — fully deterministic, no LLM call.  Stores the
    similarity score on each paper as `query_cosine` so the post-retrieval
    relevance gate can read it without re-embedding.

    Args:
        papers: papers from a category search (may contain up to retmax).
        query:  the original user query (NOT the per-category reformulation).
        n:      top-N to keep after ranking.

    Returns:
        Papers sorted descending by query_cosine, capped at n.  If fewer
        than n papers were given, returns all of them ranked.
    """
    if not papers:
        return []

    # Build per-paper text: title + abstract (truncate to keep embedding cheap)
    texts = [
        f"{p.title}\n{p.abstract[:1500]}"
        for p in papers
    ]
    try:
        vecs = embed_texts([query] + texts)
    except Exception as exc:
        logger.error(f"rank_top_n_by_query: embedding failed: {exc}")
        # Fall back to PubMed's native order (which was already 'relevance' sorted).
        return papers[:n]

    query_vec = vecs[0]
    paper_vecs = vecs[1:]

    for p, v in zip(papers, paper_vecs):
        p.query_cosine = round(cosine_similarity(query_vec, v), 4)

    ranked = sorted(papers, key=lambda p: (p.query_cosine or 0.0), reverse=True)
    return ranked[:n]


# =============================================================================
# v2.1 NEW — Post-retrieval relevance check (per-abstract cosine gate)
# =============================================================================

def check_retrieval_relevance(
    papers:    list[PubMedPaper],
    threshold: float = config.RELEVANCE_THRESHOLD_PER_ABSTRACT,
    min_pass:  int   = config.MIN_RELEVANT_ABSTRACTS,
) -> tuple[bool, int, list[PubMedPaper], float]:
    """
    Apply per-abstract cosine threshold to a category's retrieved papers.

    Pure check — assumes `papers` already carry `query_cosine` from
    rank_top_n_by_query().

    Per-abstract cosine threshold.  No
    separate mean-cosine gate — the per-abstract bar is now strict enough
    that tangential papers don't pass individually, so an aggregate check
    is redundant.

    Decision logic:
        passed = (n_passing >= min_pass)

    The reported `mean_passing` is for diagnostics in the log only; it
    doesn't gate the decision.

    Returns:
        (passed, n_passing, papers_passing, mean_cosine_of_passing_papers)
    """
    if not papers:
        return False, 0, [], 0.0

    passing = [p for p in papers if (p.query_cosine or 0.0) >= threshold]
    if passing:
        mean_cos_passing = round(
            sum((p.query_cosine or 0.0) for p in passing) / len(passing), 4
        )
    else:
        mean_cos_passing = 0.0

    passed = len(passing) >= min_pass

    logger.info(
        f"check_retrieval_relevance: {len(papers)} papers, "
        f"n_relevant={len(passing)}/{len(papers)} (threshold={threshold}), "
        f"mean_passing={mean_cos_passing:.3f} → "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed, len(passing), passing, mean_cos_passing


# =============================================================================
# Metadata extraction from PubMed abstracts  (kept from v2 — internal only)
# =============================================================================

_METADATA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract structured metadata from a scientific paper's abstract.\n"
     "Return: topic, methods, key_findings, limitations, future_recommendations.\n"
     "Each field: 1-2 sentences. Write 'Not available' if absent."),
    ("human",
     "Title: {title}\n\nAbstract: {abstract}"),
])


def extract_metadata_pubmed(
    papers:    list[PubMedPaper],
    topic:     str,
    node_name: str = "N_extract_metadata_pubmed",
) -> list[PubMedPaper]:
    """
    Extract structured metadata for each paper from its abstract.

    Kept internal in v2.1: gap analysis uses key_findings + future_recommendations
    of past/future chunks, but the user does not see a metadata tab.
    Metadata is surfaced only via the per-source ⓘ popover on the
    hypothesis card.
    """
    chain = _METADATA_PROMPT | _llm().with_structured_output(_PaperMetadataOutput)

    for paper in papers:
        if not paper.abstract:
            continue
        try:
            result: _PaperMetadataOutput = chain.invoke({
                "title":    paper.title,
                "abstract": paper.abstract[:2500],
            })
            paper.metadata = {
                "topic":                  result.topic,
                "methods":                result.methods,
                "key_findings":           result.key_findings,
                "limitations":            result.limitations,
                "future_recommendations": result.future_recommendations,
            }
            _log(node_name, config.MAIN_LLM_MODEL, "llm",
                 f"extract metadata: {paper.pmid or paper.title[:30]}",
                 paper.abstract[:500], result.key_findings)
        except Exception as exc:
            logger.warning(
                f"[{node_name}] metadata extraction failed for {paper.pmid}: {exc}"
            )
            paper.metadata = None

    n_done = sum(1 for p in papers if p.metadata)
    logger.info(f"[{node_name}] extracted metadata for {n_done}/{len(papers)} papers")
    return papers


# =============================================================================
# Contradictory evidence search (new — called from N16 after gate passes)
# =============================================================================

class _ContradictoryQueryOutput(BaseModel):
    query: str = Field(
        description=(
            "PubMed search query (5-15 words, MeSH-compatible) that finds "
            "evidence CONTRADICTING the hypothesis — opposing effects, null "
            "results, or published refutations."
        )
    )


class _ContradictoryVerificationOutput(BaseModel):
    verdict: str = Field(
        description=(
            "Exactly one of: 'contradict', 'support', 'neutral'. "
            "'contradict' = the paper presents null results, opposing findings, "
            "or directly challenges the hypothesis claim. "
            "'support' = the paper supports or is consistent with the hypothesis. "
            "'neutral' = the paper discusses the same topic but does not clearly "
            "support or contradict the specific claim."
        )
    )
    reason: str = Field(
        description="One sentence explaining the verdict."
    )


_VERIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a neuroscience literature reviewer.\n"
     "Read the hypothesis and a paper abstract. Decide whether the paper "
     "CONTRADICTS, SUPPORTS, or is NEUTRAL to the hypothesis.\n\n"
     "Rules:\n"
     "- 'contradict': the paper reports null results, an opposing relationship, "
     "  a failed replication, or directly challenges the claim in the hypothesis\n"
     "- 'support': the paper's findings are consistent with or reinforce the hypothesis\n"
     "- 'neutral': the paper is topically related but does not clearly bear on "
     "  the specific prediction\n\n"
     "Be strict: only mark 'contradict' when the paper's findings directly "
     "oppose the specific causal or correlational claim. Do not mark 'contradict' "
     "just because the paper discusses the same topic with supporting findings."),
    ("human",
     "Hypothesis: {hypothesis}\n\n"
     "Paper title: {title}\n"
     "Abstract: {abstract}"),
])


def search_contradictory_evidence(
    hypothesis_text: str,
    topic:           str,
    n:               int = 5,
    node_name:       str = "N16_contradictory",
) -> dict:
    """
    Generate a contradictory PubMed query via LLM, search, and return results.

    Returns:
        {found: bool, n_found: int, papers: list[dict], query: str}
    Fails silently — returns found=False on any error.
    """
    from src.cost_tracking import count_tokens, get_tracker

    _PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a critical neuroscience reviewer. Given a research hypothesis, "
         "write a PubMed search query that finds CONTRADICTORY evidence.\n\n"
         "Rules for the query:\n"
         "1. Target the SAME core relationship (same mechanism, same outcome type)\n"
         "2. Target the SAME or closely similar POPULATION — do NOT broaden to "
         "   unrelated diseases or conditions (e.g. if the hypothesis is about "
         "   cardiovascular risk, do NOT search Parkinson's, addiction, or trauma)\n"
         "3. Target the SAME or similar METHODS — if the hypothesis uses VBM, "
         "   search for VBM; if it uses serum biomarkers, search for those; "
         "   do NOT switch to unrelated measurement approaches\n"
         "4. Frame it as a search for NULL results, opposing effects, or studies "
         "   that challenge the specific claim\n"
         "5. 5-15 words, PubMed/MeSH-compatible\n"
         "6. Do NOT use NOT operator; instead search for the opposing finding "
         "   (e.g. 'no association', 'no significant relationship', 'inconsistent')"),
        ("human",
         "Hypothesis: {hypothesis}\nTopic context: {topic}"),
    ])

    llm = ChatOpenAI(
        model=config.MAIN_LLM_MODEL,
        temperature=0.0,
        seed=config.LLM_SEED,
    )
    chain = _PROMPT | llm.with_structured_output(_ContradictoryQueryOutput)

    try:
        result = chain.invoke({
            "hypothesis": hypothesis_text[:300],
            "topic":      topic,
        })
        query = result.query.strip()

        tracker = get_tracker()
        tracker.log_call(
            node_name=node_name,
            model=config.MAIN_LLM_MODEL,
            call_type="llm",
            summary=f"contradictory query: {query[:50]}",
            input_tokens=count_tokens(hypothesis_text[:300], config.MAIN_LLM_MODEL),
            output_tokens=count_tokens(query, config.MAIN_LLM_MODEL),
        )
        logger.info(f"[{node_name}] contradictory query: '{query}'")

        # Search PubMed — try with year filter first, then without
        safe_search = with_retries(_esearch, max_attempts=2, exceptions=(Exception,))
        pmids = safe_search(query, n, config.PUBMED_YEARS_BACK)
        if not pmids and config.PUBMED_FALLBACK_NO_YEAR:
            pmids = safe_search(query, n, None)

        if not pmids:
            logger.info(f"[{node_name}] no contradictory papers found")
            return {"found": False, "n_found": 0, "papers": [], "query": query}

        papers = _efetch(pmids[:n])

        # Relevance filter — primary: cosine to hypothesis; secondary: cosine to topic.
        #
        # The hypothesis is the correct primary reference because the search is about
        # the specific claim being made (not just the broad topic).  Contradictory
        # papers share most domain vocabulary with the hypothesis (visual cortex, UHF
        # MRI, etc.) and only differ on directional/outcome words, so they still score
        # reasonably high (0.18–0.28).  Completely off-topic papers score near zero.
        #
        # Threshold 0.18 (vs main pipeline's 0.22) — slightly permissive to avoid
        # filtering valid contradictions that use strongly opposing language.
        # Topic fallback (0.20) catches edge cases where hypothesis framing is very
        # specific but the paper is clearly on-topic.
        HYP_THRESHOLD   = 0.30   # hypothesis–abstract cosine; adjacent-domain papers score 0.15–0.22
        TOPIC_THRESHOLD = 0.30

        if papers:
            # Score against hypothesis (primary)
            hyp_ranked = rank_top_n_by_query(papers=papers, query=hypothesis_text, n=n)
            hyp_scores = {p.pmid: (p.query_cosine or 0.0) for p in hyp_ranked}

            # Score against topic (secondary)
            topic_ranked = rank_top_n_by_query(papers=papers, query=topic, n=n)
            topic_scores = {p.pmid: (p.query_cosine or 0.0) for p in topic_ranked}

            papers = [
                p for p in hyp_ranked
                if hyp_scores.get(p.pmid, 0.0) >= HYP_THRESHOLD
                or topic_scores.get(p.pmid, 0.0) >= TOPIC_THRESHOLD
            ]

        if not papers:
            logger.info(f"[{node_name}] contradictory papers filtered out (all below cosine threshold)")
            return {"found": False, "n_found": 0, "papers": [], "query": query}

        # ── Stage 2: LLM verification ──────────────────────────────────────
        # Cosine similarity only filters by topic — it cannot distinguish
        # "same topic, opposing finding" from "same topic, supporting finding".
        # Each paper's abstract is now read by an LLM judge that decides:
        # contradict / support / neutral.  Only 'contradict' papers pass.
        _verif_llm   = ChatOpenAI(
            model=config.MAIN_LLM_MODEL,
            temperature=0.0,
            seed=config.LLM_SEED,
        )
        _verif_chain = _VERIFICATION_PROMPT | _verif_llm.with_structured_output(
            _ContradictoryVerificationOutput
        )

        verified = []
        for p in papers:
            abstract = (p.abstract or "")[:1500]
            title    = p.title or ""
            if not abstract:
                continue
            try:
                verif: _ContradictoryVerificationOutput = _verif_chain.invoke({
                    "hypothesis": hypothesis_text[:400],
                    "title":      title,
                    "abstract":   abstract,
                })
                logger.debug(
                    f"[{node_name}] verify '{title[:50]}' → {verif.verdict} | {verif.reason[:80]}"
                )
                if verif.verdict.lower().strip() == "contradict":
                    verified.append(p)
            except Exception as exc:
                logger.warning(f"[{node_name}] verification failed for '{title[:40]}': {exc}")

        logger.info(
            f"[{node_name}] verification: {len(papers)} cosine-passed → "
            f"{len(verified)} confirmed contradictory"
        )

        if not verified:
            return {"found": False, "n_found": 0, "papers": [], "query": query}

        papers_dicts = [p.to_state_dict() for p in verified]
        return {
            "found":   True,
            "n_found": len(verified),
            "papers":  papers_dicts,
            "query":   query,
        }

    except Exception as exc:
        logger.warning(f"[{node_name}] contradictory evidence search failed: {exc}")
        return {"found": False, "n_found": 0, "papers": [], "query": ""}
