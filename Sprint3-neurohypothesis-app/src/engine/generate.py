"""
Hypothesis generation engine for Neurohypothesis.

Business logic for every node in the hypothesis pipeline.  LangGraph
node wrappers (which read/write AgentState) live in src/graph/nodes.py
and call these functions.

Node coverage:
    N2   parse_topic              — LLM: topic → {method, domain, focus}
    N8   select_categories        — deterministic: alphabetical complements
    N9   retrieve_evidence        — handled in nodes.py (metadata path for
                                    PubMed; vectorstore for PDFs; both for Path C)
    N10  summarize_past           — LLM: past chunks → 3 bullets
    N11  summarize_future         — LLM: future chunks → 3 bullets
    N12  compute_gap              — embedding: gap_score = 1 − cos(past, future)
    N13  generate_hypothesis      — LLM: cross-scale hypothesis
    N14  score_originality        — embedding: cosine vs past_summary
    N15  judge_plausibility       — LLM-as-judge: 6 dimensions
    N16  quality_gate             — deterministic pass/fail

Path C extension:
    integrate_hypotheses         — LLM (T=0.2) merges a PDF-side and a
                                    PubMed-side hypothesis into one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from pydantic import BaseModel, Field

import config
from src.engine.originality import OriginalityResult, score_originality_against_summary


# =============================================================================
# Pydantic output schemas
# =============================================================================

class _ParsedTopicOutput(BaseModel):
    primary_method: str = Field(
        description="Main neuroscience method implied by the topic "
                    "(e.g. 'fMRI', 'GWAS', 'electrophysiology', 'computational modelling')"
    )
    primary_domain: str = Field(
        description="Main brain system or disorder (e.g. 'hippocampus', "
                    "'Alzheimer's disease', 'working memory', 'dopamine system')"
    )
    focus: str = Field(
        description="The specific question or relationship of interest"
    )


class _BulletSummary(BaseModel):
    bullets: list[str] = Field(
        description="Exactly 5 concise bullet points, each 15-25 words",
        min_length=1,
        max_length=5,
    )


class _HypothesisLLMOutput(BaseModel):
    statement: str = Field(
        description=(
            "A single, testable research hypothesis stated as a prediction "
            "(1-2 sentences). Must not restate past findings."
        )
    )
    supported_by: list[str] = Field(
        description="List of paper IDs or PMIDs cited as supporting evidence"
    )
    suggested_approach: list[str] = Field(
        description="2-3 short bullets describing how to test the hypothesis"
    )


class _PlausibilityLLMOutput(BaseModel):
    novelty:                int        = Field(ge=1, le=5)
    testability:            int        = Field(ge=1, le=5)
    mechanistic_coherence:  int        = Field(ge=1, le=5)
    citation_traceability:  int        = Field(ge=1, le=5)
    conflict_awareness:     int        = Field(ge=1, le=5)
    usefulness:             int        = Field(ge=1, le=5)
    verdict:                str        = Field(description="One-sentence overall assessment")
    improvement_tips:       list[str]  = Field(
        default_factory=list,
        description=(
            "For each dimension scored 1 or 2, give one concrete, actionable "
            "suggestion to improve it.  Leave empty if all dimensions score ≥ 3."
        )
    )


class _IntegrationOutput(BaseModel):
    """Path C: integrated hypothesis."""
    statement: str = Field(
        description="One integrated hypothesis (1-3 sentences) "
                    "that synthesises both source hypotheses."
    )
    supported_by: list[str] = Field(
        description="Union of relevant paper IDs from both source hypotheses"
    )
    suggested_approach: list[str] = Field(
        description="2-4 bullets on how to test the integrated hypothesis"
    )


# =============================================================================
# Public result types
# =============================================================================

@dataclass
class ParsedTopic:
    primary_method: str
    primary_domain: str
    focus:          str

    def to_dict(self) -> dict[str, str]:
        return {
            "primary_method": self.primary_method,
            "primary_domain": self.primary_domain,
            "focus":          self.focus,
        }


@dataclass
class EvidenceBundle:
    """Past and future chunk texts for one hypothesis's category pair."""
    past_chunks:      list[dict[str, str]]
    future_chunks:    list[dict[str, str]]
    paper_ids:        list[str]
    categories_used:  list[str]


@dataclass
class HypothesisOutput:
    statement:          str
    supported_by:       list[str]
    suggested_approach: list[str]


@dataclass
class PlausibilityResult:
    scores:           dict[str, float]
    average:          float
    verdict:          str
    passes_gate:      bool
    improvement_tips: list[str] = field(default_factory=list)


@dataclass
class QualityGateDecision:
    passes:             bool
    failure_reason:     str
    originality_ok:     bool
    plausibility_ok:    bool
    best_of_attempts:   bool


# =============================================================================
# Shared LLM factory  (v2.1: seed and temperature both controlled)
# =============================================================================

def _llm(
    temperature: float = config.MAIN_LLM_TEMPERATURE,
    seed:        int   = config.LLM_SEED,
) -> ChatOpenAI:
    """Return a ChatOpenAI configured with the main model, temperature and seed."""
    return ChatOpenAI(
        model=config.MAIN_LLM_MODEL,
        temperature=temperature,
        seed=seed,
    )


def _log(
    node_name:    str,
    summary:      str,
    prompt_text:  str,
    output_text:  str,
    model:        str = config.MAIN_LLM_MODEL,
) -> None:
    """Log one LLM call to SessionCostTracker (best-effort)."""
    try:
        from src.cost_tracking import get_tracker, count_tokens
        get_tracker().log_call(
            node_name=node_name,
            model=model,
            call_type="llm",
            summary=summary,
            input_tokens=count_tokens(prompt_text, model),
            output_tokens=count_tokens(output_text, model),
        )
    except Exception as exc:
        logger.warning(f"Cost tracking failed in {node_name}: {exc}")


# =============================================================================
# N2 — parse_topic
# =============================================================================

_PARSE_TOPIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a neuroscience research analyst. Given a user's research topic, "
     "extract three components that will guide a structured literature search.\n\n"
     "primary_method — the dominant neuroscience METHOD implied "
     "(e.g. 'fMRI', 'GWAS', 'patch-clamp electrophysiology', 'computational modelling', "
     "'behavioural testing'). If no specific method is stated, infer the most "
     "likely method for this type of question.\n\n"
     "primary_domain — the brain system, disease, or biological level "
     "(e.g. 'hippocampal neurogenesis', 'Alzheimer's disease', "
     "'dopaminergic reward circuitry', 'synaptic plasticity').\n\n"
     "focus — the specific relationship or question being asked "
     "(e.g. 'effect of LDL cholesterol on grey matter volume', "
     "'APOE-ε4 genotype and amyloid-β deposition rate')."),
    ("human", "Research topic: {topic}"),
])


def parse_topic(
    topic:     str,
    node_name: str = "N2_parse_topic",
) -> ParsedTopic:
    """Extract {primary_method, primary_domain, focus} from raw topic."""
    chain = _PARSE_TOPIC_PROMPT | _llm().with_structured_output(_ParsedTopicOutput)
    try:
        result: _ParsedTopicOutput = chain.invoke({"topic": topic})
        _log(node_name, f"parse topic: {topic[:40]}", f"Topic: {topic}", str(result))
        logger.info(
            f"[{node_name}] method='{result.primary_method}' "
            f"domain='{result.primary_domain}' focus='{result.focus[:60]}'"
        )
        return ParsedTopic(
            primary_method=result.primary_method,
            primary_domain=result.primary_domain,
            focus=result.focus,
        )
    except Exception as exc:
        logger.error(f"[{node_name}] parse_topic LLM failed: {exc} — using fallback values")
        return ParsedTopic(
            primary_method="neuroscience research",
            primary_domain=topic[:60],
            focus=topic,
        )


# =============================================================================
# N8 — select_categories  (v2.1: alphabetical complement, no paper counts)
# =============================================================================

def order_categories(primary: str) -> list[str]:
    """
    Build the ordered category list for one query.

    [primary, comp_a, comp_b, comp_c, comp_d, comp_e]
        where comp_* are the other 5 categories from config.CATEGORIES
        in alphabetical order (skipping the primary).

    This is the deterministic ordering used by H1..H6:
        H1 → primary alone
        H2 → primary + ordered[1]
        H3 → primary + ordered[2]
        H4 → primary + ordered[3]
        H5 → primary + ordered[4]
        H6 → primary + ordered[5]
    """
    others = sorted(c for c in config.CATEGORIES if c != primary)
    return [primary] + others


def select_categories(
    ordered:    list[str],
    hyp_index:  int,
) -> tuple[str, list[str]]:
    """
    Pick the primary + complementary category pair for hypothesis index `hyp_index`.

    Returns:
        (primary_category, [complementary_categories])
        H1 returns empty complement list; H2..H6 return one category.
    """
    if not ordered:
        return "Behavioral & Cognitive Neuroscience", []

    primary = ordered[0]

    # H1: primary alone
    if hyp_index == 0:
        return primary, []

    # H2..H6: primary + ordered[hyp_index]
    if hyp_index < len(ordered):
        return primary, [ordered[hyp_index]]

    # Beyond H6: wrap around (shouldn't happen with MAX_HYPOTHESES=6 and 6 cats)
    return primary, [ordered[1]] if len(ordered) > 1 else []


# =============================================================================
# N10 — summarize_past
# =============================================================================

_PAST_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are summarising PAST research findings across multiple neuroscience "
     "papers. Given past-tagged content (results, established findings, "
     "demonstrated effects), produce exactly 5 concise bullet points capturing:\n"
     "  • the main established findings on this topic\n"
     "  • the methods used and populations studied\n"
     "  • the consistency/conflicts across the studies\n\n"
     "Rules:\n"
     "- Exactly 5 bullets, each 15-25 words\n"
     "- Synthesise across papers — not one bullet per paper\n"
     "- Focus on what HAS been done\n"
     "- Stay focused on the topic: {topic}"),
    ("human", "Past-tagged chunks:\n{chunks_text}"),
])


def summarize_past(
    evidence:  EvidenceBundle,
    topic:     str,
    node_name: str = "N10_summarize_past",
) -> list[str]:
    """Summarise past-tagged evidence into 3 bullets at T=0."""
    if not evidence.past_chunks:
        logger.warning(f"[{node_name}] No past chunks — returning placeholder")
        return ["No past-tagged content available for the selected categories."]

    chunks_text = "\n\n".join(
        f"[{c['paper_id']}] {c['text']}" for c in evidence.past_chunks[:config.SUMMARY_MAX_CHUNKS]
    )
    prompt_text = f"Topic: {topic}\n\nChunks:\n{chunks_text[:3000]}"
    chain = _PAST_SUMMARY_PROMPT | _llm().with_structured_output(_BulletSummary)

    try:
        result: _BulletSummary = chain.invoke({"topic": topic, "chunks_text": chunks_text})
        bullets = result.bullets[:3]
        _log(node_name, f"summarize past: {topic[:40]}", prompt_text, " | ".join(bullets))
        logger.info(f"[{node_name}] {len(bullets)} past bullets generated")
        return bullets
    except Exception as exc:
        logger.error(f"[{node_name}] summarize_past failed: {exc}")
        return [f"Past summarisation unavailable: {exc}"]


# =============================================================================
# N11 — summarize_future
# =============================================================================

_FUTURE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are summarising FUTURE research directions across multiple neuroscience "
     "papers. Given future-tagged content (limitations, recommendations, open "
     "questions), produce exactly 5 concise bullet points capturing:\n"
     "  • the main methodological limitations to address\n"
     "  • the specific gaps and unresolved questions\n"
     "  • the recommended future study designs\n\n"
     "Rules:\n"
     "- Exactly 5 bullets, each 15-25 words\n"
     "- Synthesise across papers — not one bullet per paper\n"
     "- Focus on what SHOULD be studied or REMAINS unknown\n"
     "- Stay focused on the topic: {topic}"),
    ("human", "Future-tagged chunks:\n{chunks_text}"),
])


def summarize_future(
    evidence:  EvidenceBundle,
    topic:     str,
    node_name: str = "N11_summarize_future",
) -> list[str]:
    """Summarise future-tagged evidence into 3 bullets at T=0."""
    if not evidence.future_chunks:
        logger.warning(f"[{node_name}] No future chunks — returning placeholder")
        return ["No future-tagged content available for the selected categories."]

    chunks_text = "\n\n".join(
        f"[{c['paper_id']}] {c['text']}" for c in evidence.future_chunks[:config.SUMMARY_MAX_CHUNKS]
    )
    prompt_text = f"Topic: {topic}\n\nChunks:\n{chunks_text[:3000]}"
    chain = _FUTURE_SUMMARY_PROMPT | _llm().with_structured_output(_BulletSummary)

    try:
        result: _BulletSummary = chain.invoke({"topic": topic, "chunks_text": chunks_text})
        bullets = result.bullets[:3]
        _log(node_name, f"summarize future: {topic[:40]}", prompt_text, " | ".join(bullets))
        logger.info(f"[{node_name}] {len(bullets)} future bullets generated")
        return bullets
    except Exception as exc:
        logger.error(f"[{node_name}] summarize_future failed: {exc}")
        return [f"Future summarisation unavailable: {exc}"]


# =============================================================================
# N12 — compute_gap
# =============================================================================

def compute_gap(past_summary: str, future_summary: str) -> float:
    """gap_score = 1 − cosine(embed(past_summary), embed(future_summary))."""
    from src.utils import cosine_similarity, embed_texts

    if not past_summary.strip() or not future_summary.strip():
        logger.warning("[N12_compute_gap] Empty summary — returning neutral gap 0.5")
        return 0.5

    try:
        vecs = embed_texts([past_summary, future_summary])
        sim  = cosine_similarity(vecs[0], vecs[1])
        return round(1.0 - sim, 4)
    except Exception as exc:
        logger.error(f"[N12_compute_gap] Embedding failed: {exc} — returning 0.5")
        return 0.5


# =============================================================================
# N13 — generate_hypothesis  (v2.1: previous_statements + per-H seed)
# =============================================================================

_HYPOTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a neuroscience research analyst generating a novel, testable "
     "research hypothesis.\n\n"
     "{categories_block}"
     "NEUROSCIENCE METHOD-OUTCOME CONSISTENCY (mandatory):\n"
     "Match every method to its correct outcome — violations will be flagged:\n"
     "- DTI / DWI / diffusion MRI → white matter microstructure (FA, MD, "
     "  tractography, WM tracts).  Do NOT use DTI to measure grey matter.\n"
     "- VBM / cortical thickness / grey matter density → grey matter morphology.  "
     "  Do NOT apply these to white matter.\n"
     "- fMRI BOLD → functional connectivity or task activation (NOT structural)\n"
     "- PET → receptor binding, metabolism, amyloid/tau deposition\n"
     "- EEG / MEG → oscillations, ERPs, temporal dynamics (NOT morphology)\n"
     "- MRS → metabolite concentrations (GABA, glutamate, NAA)\n\n"
     "Requirements:\n"
     "- 1-2 sentences — a specific, testable prediction\n"
     "- Must NOT restate any past finding verbatim\n"
     "- Must be genuinely inspired by the GAP between past and future summaries\n"
     "- Must propose a NEW variable, population, method, or mechanism\n"
     "- Must include concrete neuroscience terminology (brain regions, "
     "  biomarkers, populations, methods)\n"
     "- Must stay focused on the topic: {topic}\n\n"
     "{previous_block}"
     "{failure_block}"
     "Also provide:\n"
     "- supported_by: list of paper IDs from the evidence that relate to this hypothesis\n"
     "- suggested_approach: 2-3 short bullets describing how to test it"),
    ("human",
     "TOPIC: {topic}\n\n"
     "PAST SUMMARY (what has been done):\n{past_summary}\n\n"
     "FUTURE SUMMARY (what is recommended):\n{future_summary}\n\n"
     "GAP SCORE: {gap_score:.3f} (higher = bigger gap, more room for novelty)\n\n"
     "AVAILABLE PAPER IDs: {paper_ids}"),
])

_FAILURE_BLOCK_TEMPLATE = (
    "PREVIOUS ATTEMPT FAILED QUALITY GATE (attempt {attempt}):\n"
    "Reason: {reason}\n"
    "Improvement required: generate a MORE ORIGINAL and SCIENTIFICALLY SOUND "
    "hypothesis that addresses the failure reason above.\n\n"
)

_PREVIOUS_BLOCK_TEMPLATE = (
    "PREVIOUSLY GENERATED HYPOTHESES IN THIS SESSION (do NOT repeat or paraphrase):\n"
    "{previous_text}\n\n"
    "Your hypothesis must propose a DIFFERENT mechanism, variable, population, "
    "or angle than each of these.\n\n"
)


def _categories_block(primary_cat: str, comp_cats: list[str]) -> str:
    """
    Build the categories instruction block for the hypothesis prompt.

    H1 (no complement) → stays in the primary category.
    H2..H6 (with complement) → mandates genuine cross-scale integration
    with citations from BOTH categories.
    """
    if not comp_cats:
        return (
            f"PRIMARY CATEGORY (mandatory anchor): {primary_cat}.\n"
            "Your hypothesis should be directly answerable using methods from "
            "this primary category.\n\n"
        )
    cats = [primary_cat] + comp_cats
    return (
        "CROSS-SCALE INTEGRATION (mandatory):\n"
        f"  Primary category:       {primary_cat}\n"
        f"  Complementary category: {', '.join(comp_cats)}\n\n"
        "Your hypothesis MUST:\n"
        "1. Be grounded in the primary category — name a specific method or "
        f"   finding from {primary_cat}.\n"
        "2. Explicitly incorporate a mechanism, variable, or finding from the "
        f"   complementary category ({', '.join(comp_cats)}).\n"
        "3. Reference paper IDs from BOTH categories in supported_by — a "
        "   hypothesis citing only the primary category is incomplete.\n"
        "4. Make the cross-scale CONNECTION the core novelty: explain HOW the "
        "   two levels of analysis inform each other.\n\n"
        "A hypothesis that could have been generated from the primary category "
        "alone — without the complementary evidence — does NOT meet the brief.\n\n"
    )


def generate_hypothesis(
    topic:                str,
    primary_cat:          str,
    comp_cats:            list[str],
    past_summary:         str,
    future_summary:       str,
    gap_score:            float,
    paper_ids:            list[str],
    failure_reason:       str | None       = None,
    attempt:              int              = 0,
    hyp_index:            int              = 0,
    previous_statements:  list[str] | None = None,
    node_name:            str              = "N13_generate_hypothesis",
) -> HypothesisOutput:
    """
    Generate one hypothesis from the evidence summaries and category combo.

    v2.1 additions:
    - `hyp_index`: used as a seed offset so each of H1..H6 is reproducible
      but distinct under temperature 0.
    - `previous_statements`: prior hypotheses already generated this session;
      injected into the prompt so the LLM avoids repeating them.
    - Temperature: config.GEN_LLM_TEMPERATURE (0.0 by default; bump to 0.2
      if outputs feel derivative across runs).
    """
    categories_block = _categories_block(primary_cat, comp_cats)
    failure_block    = (
        _FAILURE_BLOCK_TEMPLATE.format(attempt=attempt, reason=failure_reason)
        if failure_reason and attempt > 0 else ""
    )
    previous_block = ""
    if previous_statements:
        prev_text = "\n".join(
            f"  H{i+1}: {s[:200]}" for i, s in enumerate(previous_statements)
        )
        previous_block = _PREVIOUS_BLOCK_TEMPLATE.format(previous_text=prev_text)

    past_text   = "\n".join(f"• {b}" for b in past_summary.split("\n")   if b.strip())
    future_text = "\n".join(f"• {b}" for b in future_summary.split("\n") if b.strip())
    prompt_text = (
        f"TOPIC: {topic}\nPAST: {past_text[:400]}\nFUTURE: {future_text[:400]}\n"
        f"GAP: {gap_score:.3f}\nCATEGORIES: {primary_cat}+{','.join(comp_cats)}"
    )

    # Seed offset per hypothesis index → reproducible across runs, distinct across H1..H6.
    seed = config.LLM_SEED + hyp_index
    chain = (
        _HYPOTHESIS_PROMPT
        | _llm(temperature=config.GEN_LLM_TEMPERATURE, seed=seed)
            .with_structured_output(_HypothesisLLMOutput)
    )

    try:
        result: _HypothesisLLMOutput = chain.invoke({
            "topic":            topic,
            "categories_block": categories_block,
            "past_summary":     past_text,
            "future_summary":   future_text,
            "gap_score":        gap_score,
            "paper_ids":        ", ".join(paper_ids[:20]),
            "previous_block":   previous_block,
            "failure_block":    failure_block,
        })
        _log(node_name, f"generate H{hyp_index+1}: {topic[:40]}", prompt_text, result.statement)
        logger.info(
            f"[{node_name}] H{hyp_index+1} attempt={attempt+1} seed={seed} | "
            f"{result.statement[:100]}…"
        )
        return HypothesisOutput(
            statement=result.statement,
            supported_by=result.supported_by,
            suggested_approach=result.suggested_approach,
        )
    except Exception as exc:
        logger.error(f"[{node_name}] generate_hypothesis failed (H{hyp_index+1}): {exc}")
        return HypothesisOutput(
            statement=f"Hypothesis generation failed: {exc}",
            supported_by=[],
            suggested_approach=[],
        )


# =============================================================================
# Path A — PDF-only incremental generation (one hypothesis per anchored gap pair)
# =============================================================================

class _PathAItem(BaseModel):
    statement:          str       = Field(description="One testable hypothesis (1-2 sentences), a forward-looking prediction, not a restatement of past findings")
    supported_by:       list[str] = Field(description="Paper IDs from the provided list that support this hypothesis")
    suggested_approach: list[str] = Field(description="2-3 short bullets on how to test it")


_PATH_A_ONE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Neuroscience research assistant. Generate ONE testable hypothesis grounded "
     "in the user's OWN uploaded papers, anchored on this specific research gap:\n"
     "  PAST (established): {past_anchor}\n"
     "  FUTURE (open question / limitation): {future_anchor}\n"
     "Rules:\n"
     "- A forward-looking prediction, not a restatement of the past finding.\n"
     "- Exploit the gap between the anchor's past and future.\n"
     "- Ground it in the provided paper IDs via supported_by.\n"
     "- It MUST be clearly different from any previous hypotheses listed below.\n"
     "{previous_block}"),
    ("human",
     "TOPIC: {topic}\n\nBACKGROUND PAST:\n{past_summary}\n\n"
     "BACKGROUND FUTURE:\n{future_summary}\n\nPAPER IDS: {paper_ids}"),
])


def generate_path_a_one(
    topic:               str,
    past_anchor:         str,
    future_anchor:       str,
    past_summary:        str,
    future_summary:      str,
    gap_score:           float,
    paper_ids:           list[str],
    previous_statements: list[str] | None = None,
    seed_offset:         int = 0,
    node_name:           str = "N_a_generate",
) -> HypothesisOutput:
    """Generate ONE hypothesis anchored on a specific past->future gap pair."""
    past_text   = "\n".join(f"• {b}" for b in past_summary.split("\n")   if b.strip())
    future_text = "\n".join(f"• {b}" for b in future_summary.split("\n") if b.strip())
    previous_block = ""
    if previous_statements:
        prev = "\n".join(f"  - {s[:160]}" for s in previous_statements)
        previous_block = f"PREVIOUS HYPOTHESES (be clearly different from these):\n{prev}"

    chain = (
        _PATH_A_ONE_PROMPT
        | _llm(temperature=config.PATH_A_GEN_TEMPERATURE, seed=config.LLM_SEED + seed_offset)
            .with_structured_output(_PathAItem)
    )
    try:
        r: _PathAItem = chain.invoke({
            "topic":          topic,
            "past_anchor":    past_anchor[:400],
            "future_anchor":  future_anchor[:400],
            "past_summary":   past_text[:1000],
            "future_summary": future_text[:1000],
            "paper_ids":      ", ".join(paper_ids[:20]),
            "previous_block": previous_block,
        })
        _log(node_name, f"path A one: {topic[:40]}", topic, r.statement)
        logger.info(f"[{node_name}] generated (seed_offset={seed_offset}): {r.statement[:90]}…")
        return HypothesisOutput(
            statement=r.statement,
            supported_by=r.supported_by,
            suggested_approach=r.suggested_approach,
        )
    except Exception as exc:
        logger.error(f"[{node_name}] path A generation failed: {exc}")
        return HypothesisOutput(statement=f"Hypothesis generation failed: {exc}",
                                supported_by=[], suggested_approach=[])


# =============================================================================
# N14 — score_originality  (delegates to engine/originality.py)
# =============================================================================

def score_originality(
    hypothesis_text: str,
    past_summary:    str,
    embeddings:      OpenAIEmbeddings,
    node_name:       str = "N14_score_originality",
) -> OriginalityResult:
    """
    Score how original the hypothesis is versus the past_summary.

    1 − cosine(hypothesis, past_summary). Pure math, no LLM call.
    Wraps the batch function `score_originality_against_summary` so the
    caller can pass a single hypothesis string.
    """
    hyp_dict = [{"id": "H", "statement": hypothesis_text}]
    results  = score_originality_against_summary(hyp_dict, past_summary, embeddings)

    if results:
        r = results[0]
        logger.info(
            f"[{node_name}] originality_score={r.originality_score:.4f} "
            f"grade={r.grade} passes={r.passes_gate}"
        )
        return r

    # Fallback — embedding must have failed
    from src.engine.originality import _make_result
    fallback = _make_result(
        {"id": "H", "statement": hypothesis_text},
        similarity=0.5,
    )
    logger.warning(f"[{node_name}] score_originality returned no results — using fallback")
    return fallback


# =============================================================================
# N15 — judge_plausibility
# =============================================================================

_PLAUSIBILITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a sceptical neuroscience reviewer grading a hypothesis.  "
     "Score each dimension 1 (poor) to 5 (excellent) using the rubric.  "
     "Be critical — most hypotheses do not deserve 5s.  Score below 3 "
     "whenever you find specific problems.\n\n"
     "1. novelty — Does the hypothesis genuinely go beyond what the cited "
     "papers state?  Score low if it just restates known findings.\n"
     "2. testability — Is the proposed design EXPERIMENTALLY VIABLE and "
     "INTERNALLY CONSISTENT?  Score 2 or below if the hypothesis combines "
     "incompatible methodologies.  Specific incompatibilities to flag:\n"
     "   * ex-vivo / postmortem tissue work combined with longitudinal "
     "     follow-up in living subjects (impossible — tissue is removed once)\n"
     "   * single-neuron electrophysiology combined with whole-brain fMRI "
     "     in the same subjects (not co-acquirable in humans)\n"
     "   * cross-sectional design described as longitudinal (or vice versa)\n"
     "   * animal model with human-specific behavioural readouts the animal "
     "     cannot perform\n"
     "   * postmortem measurement of behaviour or cognition (impossible)\n"
     "   * DTI / DWI / diffusion MRI used to measure GREY MATTER (DTI "
     "     measures white matter microstructure — FA, MD, tractography; grey "
     "     matter requires VBM, cortical thickness, or grey matter density)\n"
     "   * VBM / cortical thickness used to characterise WHITE MATTER tracts\n"
     "   * fMRI BOLD signal interpreted as a structural or anatomical change\n"
     "   * PET tracer claimed to measure something outside its known binding target\n"
     "3. mechanistic_coherence — Is the proposed biological mechanism plausible "
     "given established physiology and anatomy?  Score low when the proposed "
     "mechanism contradicts known biology, or when the hypothesis names "
     "biomarkers / regions that are not implicated in the system under study.\n"
     "4. citation_traceability — Can the claims be traced back to the cited "
     "papers?  Score low if the supporting paper IDs do not actually support "
     "the specific prediction.\n"
     "5. conflict_awareness — Does the hypothesis acknowledge or accommodate "
     "conflicting evidence in the field?  Score low if it ignores known "
     "contradictory findings.\n"
     "6. usefulness — If true, would it meaningfully advance the field, or "
     "is the question marginal?\n\n"
     "Return all 6 scores plus a one-sentence verdict naming any logical "
     "inconsistencies found in dimensions 2 or 3.  "
     "For every dimension scored 1 or 2, also return one concrete, actionable "
     "improvement_tip (e.g. 'Replace DTI with VBM to correctly measure grey "
     "matter volume').  Leave improvement_tips empty if all scores ≥ 3."),
    ("human",
     "Hypothesis: {statement}\n\n"
     "Supporting paper IDs: {supported_by}\n\n"
     "Topic: {topic}"),
])


def judge_plausibility(
    hypothesis_text: str,
    paper_ids:       list[str],
    topic:           str,
    node_name:       str = "N15_judge_plausibility",
) -> PlausibilityResult:
    """LLM-as-judge across 6 scientific dimensions. T=0, seeded."""
    prompt_text = f"Hypothesis: {hypothesis_text[:300]}\nTopic: {topic}"
    chain = (
        _PLAUSIBILITY_PROMPT
        | _llm(temperature=config.JUDGE_LLM_TEMPERATURE)
            .with_structured_output(_PlausibilityLLMOutput)
    )
    _DIMS = [
        "novelty", "testability", "mechanistic_coherence",
        "citation_traceability", "conflict_awareness", "usefulness",
    ]

    try:
        result: _PlausibilityLLMOutput = chain.invoke({
            "statement":    hypothesis_text,
            "supported_by": ", ".join(paper_ids[:10]) if paper_ids else "none",
            "topic":        topic,
        })
        scores  = {d: float(getattr(result, d)) for d in _DIMS}
        average = round(sum(scores.values()) / len(scores), 2)
        _log(node_name, f"plausibility judge: {hypothesis_text[:40]}",
             prompt_text, result.verdict)
        logger.info(
            f"[{node_name}] avg={average:.2f} | "
            + " ".join(f"{d[:4]}={int(scores[d])}" for d in _DIMS)
        )
        return PlausibilityResult(
            scores=scores,
            average=average,
            verdict=result.verdict,
            passes_gate=(average >= config.PLAUSIBILITY_PASS_AVG),
            improvement_tips=result.improvement_tips,
        )
    except Exception as exc:
        logger.error(f"[{node_name}] judge_plausibility failed: {exc} — fallback 3.0")
        scores = {d: 3.0 for d in _DIMS}
        return PlausibilityResult(
            scores=scores,
            average=3.0,
            verdict="Plausibility scoring unavailable.",
            passes_gate=False,
            improvement_tips=[],
        )


# =============================================================================
# N16 — quality_gate
# =============================================================================

def quality_gate(
    originality:  OriginalityResult,
    plausibility: PlausibilityResult,
    attempt:      int,
) -> QualityGateDecision:
    """Pass/fail + failure reason (deterministic, no LLM)."""
    orig_ok  = originality.passes_gate
    plaus_ok = plausibility.passes_gate
    passes   = orig_ok and plaus_ok

    if passes:
        return QualityGateDecision(
            passes=True,
            failure_reason="",
            originality_ok=True,
            plausibility_ok=True,
            best_of_attempts=False,
        )

    reasons: list[str] = []
    if not orig_ok:
        reasons.append(
            f"originality too low ({originality.originality_score:.2f} < "
            f"{config.ORIGINALITY_PASS_THRESHOLD}) — the hypothesis is too "
            f"similar to existing work ({originality.grade_label}). "
            "Propose a more novel variable, population, or mechanism."
        )
    if not plaus_ok:
        low_dims = [
            dim for dim, score in plausibility.scores.items() if score < 3
        ]
        reasons.append(
            f"scientific quality too low (avg {plausibility.average:.2f} < "
            f"{config.PLAUSIBILITY_PASS_AVG}). "
            f"Weak dimensions: {', '.join(low_dims) if low_dims else 'overall'}. "
            "Make the hypothesis more testable and mechanistically coherent."
        )
    failure_reason = " | ".join(reasons)

    exhausted = attempt + 1 >= config.QUALITY_GATE_MAX_ATTEMPTS
    if exhausted:
        logger.warning(
            f"[N16_quality_gate] EXHAUSTED after {attempt+1} attempts"
        )
        return QualityGateDecision(
            passes=True,
            failure_reason=failure_reason,
            originality_ok=orig_ok,
            plausibility_ok=plaus_ok,
            best_of_attempts=True,
        )

    return QualityGateDecision(
        passes=False,
        failure_reason=failure_reason,
        originality_ok=orig_ok,
        plausibility_ok=plaus_ok,
        best_of_attempts=False,
    )


# =============================================================================
# Path C — Integrate one PDF-derived + one PubMed-derived hypothesis
# =============================================================================

_INTEGRATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are integrating two scientific hypotheses into one cohesive, "
     "testable hypothesis.  The two source hypotheses are on the same "
     "research topic but come from different evidence bases (one from "
     "the user's uploaded papers, the other from PubMed literature).\n\n"
     "Your integration must:\n"
     "- Preserve the testable structure of BOTH source hypotheses\n"
     "- Synthesise into a single 1-3 sentence statement\n"
     "- Be more specific or mechanistically richer than either source\n"
     "- Stay grounded — do not invent new findings\n"
     "- Use concrete neuroscience terminology\n\n"
     "Topic: {topic}"),
    ("human",
     "SOURCE A (from user's uploaded papers):\n"
     "{statement_a}\n"
     "Supporting evidence: {sources_a}\n\n"
     "SOURCE B (from PubMed search):\n"
     "{statement_b}\n"
     "Supporting evidence: {sources_b}"),
])


def integrate_hypotheses(
    topic:        str,
    statement_a:  str,
    sources_a:    list[str],
    statement_b:  str,
    sources_b:    list[str],
    seed:         int = config.LLM_SEED,
    node_name:    str = "N_integrate_combined",
) -> HypothesisOutput:
    """
    Path C: LLM merges two hypotheses into one integrated hypothesis.

    Temperature = config.INTEGRATION_LLM_TEMPERATURE (0.2): enough creativity
    to find the synthesis, faithful to the sources.
    """
    chain = (
        _INTEGRATE_PROMPT
        | _llm(temperature=config.INTEGRATION_LLM_TEMPERATURE, seed=seed)
            .with_structured_output(_IntegrationOutput)
    )
    prompt_text = (
        f"Topic: {topic}\nA: {statement_a[:200]}\nB: {statement_b[:200]}"
    )
    try:
        result: _IntegrationOutput = chain.invoke({
            "topic":       topic,
            "statement_a": statement_a,
            "sources_a":   ", ".join(sources_a[:10]) if sources_a else "none",
            "statement_b": statement_b,
            "sources_b":   ", ".join(sources_b[:10]) if sources_b else "none",
        })
        _log(node_name, f"integrate: {topic[:40]}", prompt_text, result.statement)
        logger.info(f"[{node_name}] integrated: {result.statement[:120]}…")
        return HypothesisOutput(
            statement=result.statement,
            supported_by=result.supported_by,
            suggested_approach=result.suggested_approach,
        )
    except Exception as exc:
        logger.error(f"[{node_name}] integration failed: {exc}")
        # Fallback: return source A unchanged
        return HypothesisOutput(
            statement=statement_a,
            supported_by=sources_a + sources_b,
            suggested_approach=["Integration step unavailable; review both source hypotheses."],
        )
