"""
Central configuration for the Hypothesis Generator.

All magic numbers, model names, paths, and limits live here so they can
be tuned in one place without hunting through the codebase.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()

DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
EXTRACTED_JSON_DIR = DATA_DIR / "extracted_json"
EVAL_SETS_DIR = DATA_DIR / "eval_sets"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "neuroscience_papers"

EXPORTS_DIR = PROJECT_ROOT / "exports"

# =============================================================================
# Models
# =============================================================================

MAIN_LLM_MODEL = "gpt-4o-mini"
MAIN_LLM_TEMPERATURE = 0.2

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# =============================================================================
# Chunking
# =============================================================================

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

SECTION_TYPES = ["abstract", "discussion", "limitations_future"]

# =============================================================================
# Retrieval
# =============================================================================

SEMANTIC_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 6

RRF_K = 60

# =============================================================================
# Query translation
# =============================================================================

NUM_QUERY_VARIANTS = 3

# =============================================================================
# Input validation
# =============================================================================

MAX_QUERY_LENGTH = 500
MIN_QUERY_LENGTH = 3

# =============================================================================
# Cost tracking (USD per 1M tokens)
# =============================================================================

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-mini":  {"input": 0.25, "output": 2.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}

# =============================================================================
# UI
# =============================================================================

APP_TITLE = "Hypothesis Generator"
APP_SUBTITLE = "Neuroscience research assistant - RAG + PubMed + novel hypothesis generation"
DISCLAIMER = (
    "AI-generated hypotheses are research suggestions, not scientific facts. "
    "All outputs require expert validation before use in actual research."
)

# =============================================================================
# Past / Future temporal tagging (gap analysis foundation)
#
# Cosine similarity between each chunk and temporal reference descriptions.
# Used to pre-tag chunks so the gap analysis can quantitatively separate
# what has been done (past) from what is recommended (future).
# =============================================================================

TEMPORAL_REFERENCES = {
    "past": (
        "Previous research findings, established results, what has been "
        "studied, observed correlations, demonstrated effects, existing "
        "evidence, prior investigations showed, it was found that, "
        "studies have demonstrated, research has established, "
        "meta-analysis showed, systematic review found, pooled analysis "
        "revealed, odds ratio, hazard ratio, confidence interval, "
        "statistically significant association, results indicated, "
        "data showed, cohort study demonstrated, participants exhibited, "
        "the study aimed to, we hypothesised that, this study investigated, "
        "the objective was to determine, background, introduction, "
        "prior studies have shown, evidence suggests, literature indicates"
    ),
    "future": (
        "Study limitations, recommended future research, unresolved "
        "questions, methodological improvements needed, suggested "
        "investigations, knowledge gaps to address, further studies "
        "should examine, it remains to be determined, future work "
        "is needed to clarify"
    ),
}

# If the difference between past and future similarity scores is less
# than this value, the chunk is tagged as "neutral" (ambiguous content
# that contains elements of both past findings and future directions).
TEMPORAL_NEUTRAL_MARGIN = 0.1

# =============================================================================
# Past-Future gap filter (Filter 1 — pre-generation)
#
# Cosine similarity between past-tagged and future-tagged chunks.
# Pairs with similarity ABOVE this threshold are considered "not a real
# gap" — the future recommendation is too similar to what was already
# done, so generating a hypothesis from that pair would produce
# something unoriginal.
# =============================================================================

# =============================================================================
# Gap filter and originality thresholds — REPLACED by three-category grading.
# The 0.5 single threshold is superseded by VERY_ORIGINAL_THRESHOLD (0.3)
# and LESS_ORIGINAL_THRESHOLD (0.8) defined below.
# Kept here as fallback references only.
# =============================================================================

GAP_SIMILARITY_THRESHOLD = 0.8    # now same as LESS_ORIGINAL_THRESHOLD
ORIGINALITY_THRESHOLD = 0.8       # now same as LESS_ORIGINAL_THRESHOLD

# =============================================================================
# Domain keywords (for BM25 relevance and future query expansion)
# =============================================================================

DOMAIN_KEYWORDS = [
    "LDL-cholesterol", "low-density lipoprotein",
    "brain atrophy", "positive correlation", "negative correlation",
    "Alzheimer's", "Parkinson's",
    "neurodegeneration", "neuroinflammation",
    "cognitive impairment", "cognitive function",
    "grey matter volume", "white matter microstructure",
    "VBM", "TBSS", "MRI",
    "age", "sex", "cardiovascular",
    "HDL-cholesterol", "amyloid-beta", "p-tau", "alpha-synuclein",
]

# =============================================================================
# Three-category grading thresholds (applied to originality score and
# literature gap score)
#
# Grade is based on cosine similarity:
#   similarity <= VERY_ORIGINAL_THRESHOLD      → "Very original" / "Strong gap"
#   VERY_ORIGINAL_THRESHOLD < sim < LESS_ORIGINAL_THRESHOLD → "Moderately original" / "Moderate gap"
#   similarity >= LESS_ORIGINAL_THRESHOLD      → "Less original" / "Weak gap"
# =============================================================================

VERY_ORIGINAL_THRESHOLD = 0.3
LESS_ORIGINAL_THRESHOLD = 0.8

# Unified blue-tone palette for ALL graded scores:
# originality (local library), literature gap, and PubMed check.
# One palette, consistent across the whole app.
BLUE_GRADE_COLORS = {
    "very":     "#1a4f8a",   # dark blue — Very original / Strong gap
    "moderate": "#2e7bc4",   # medium blue — Moderately original / Moderate gap
    "less":     "#7ab3e0",   # lighter blue — Less original / Weak gap
}

# Keep old names as aliases so nothing breaks in originality.py
ORIGINALITY_COLORS = BLUE_GRADE_COLORS
GAP_COLORS = BLUE_GRADE_COLORS
PUBMED_COLORS = BLUE_GRADE_COLORS

# =============================================================================
# PubMed freshness check (Tool 4 — post-generation)
#
# After the hypothesis is generated and scored locally, a PubMed search
# is run to check if the hypothesis matches recent published work that
# is not in the user's local library. The same three-category grading
# is applied with a purple-tone palette (visually distinct from the
# green originality grade).
# =============================================================================

PUBMED_YEARS_BACK = 5          # Search papers from this many years back (2020+)
PUBMED_TOP_N = 5               # Number of most relevant abstracts to compare against
PUBMED_SHOW_MATCHES_AT = 0.8   # Show matching papers when similarity >= this
