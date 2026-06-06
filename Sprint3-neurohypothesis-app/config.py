"""
Central configuration for Neurohypothesis — Neuroscience Hypothesis Generation Agent.

All constants, thresholds, model names, paths, and lookup dictionaries live
here so they can be tuned in one place without hunting through the codebase.

Public API (imported by all modules):
    Paths:                  PROJECT_ROOT, DATA_DIR, UPLOADS_DIR, CHROMA_DIR,
                            EXPORTS_DIR, SQLITE_PATH, LOG_FILE
    Models:                 MAIN_LLM_MODEL, JUDGE_LLM_MODEL, EMBEDDING_MODEL,
                            MAIN_LLM_TEMPERATURE, GEN_LLM_TEMPERATURE,
                            JUDGE_LLM_TEMPERATURE, INTEGRATION_LLM_TEMPERATURE,
                            EMBEDDING_DIMENSIONS, LLM_SEED
    Pricing:                PRICING  (USD per 1M tokens)
    Chunking:               CHUNK_SIZE, CHUNK_OVERLAP
    Retrieval:              SEMANTIC_TOP_K, BM25_TOP_K, FINAL_TOP_K, RRF_K
    PubMed:                 PUBMED_PER_CATEGORY_N, PUBMED_MAX_RETRIES,
                            PUBMED_TIMEOUT_S, PUBMED_YEARS_BACK,
                            PUBMED_FALLBACK_NO_YEAR, PREDATORY_PUBLISHERS,
                            PUBMED_CHECK_YEARS_BACK, PUBMED_CHECK_TOP_N
    Per-category loop:      REFORMULATE_TEMP_ESCALATION, REFORMULATE_MAX_ATTEMPTS,
                            RELEVANCE_THRESHOLD_PER_ABSTRACT, MIN_RELEVANT_ABSTRACTS
    Session / PDF limits:   MAX_PDF_COUNT, MAX_PDF_SIZE_MB, MAX_QUERY_LENGTH,
                            MIN_QUERY_LENGTH, MAX_HYPOTHESES
    Quality gate:           ORIGINALITY_PASS_THRESHOLD, PLAUSIBILITY_PASS_AVG,
                            QUALITY_GATE_MAX_ATTEMPTS
    Originality grading:    VERY_ORIGINAL_THRESHOLD, LESS_ORIGINAL_THRESHOLD,
                            BLUE_GRADE_COLORS
    Temporal tagging:       TEMPORAL_REFERENCES, TEMPORAL_NEUTRAL_MARGIN
    Categorisation:         CATEGORIES, CATEGORY_DESCRIPTIONS,
                            SUBFIELD_TO_CATEGORY, MESH_TO_CATEGORY
    Paths / routing:        PATH_OPTIONS
    UI:                     APP_TITLE, APP_SUBTITLE, DISCLAIMER
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud: bridge st.secrets → os.environ so load_dotenv()-based
# code works unchanged on both local (.env) and cloud (Streamlit secrets).
try:
    import streamlit as st
    for _k in ["OPENAI_API_KEY", "NCBI_API_KEY", "PUBMED_EMAIL",
               "SUPABASE_URL", "SUPABASE_SERVICE_KEY"]:
        if _k not in os.environ:
            _v = st.secrets.get(_k)
            if _v:
                os.environ[_k] = _v
except Exception:
    pass  # not running in Streamlit, or secrets not configured yet

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()

DATA_DIR      = PROJECT_ROOT / "data"
UPLOADS_DIR   = DATA_DIR / "uploads"
EVAL_SETS_DIR = DATA_DIR / "eval_sets"

CHROMA_DIR  = PROJECT_ROOT / Path(os.getenv("CHROMA_DIR",  "chroma_db"))
EXPORTS_DIR = PROJECT_ROOT / Path(os.getenv("EXPORTS_DIR", "exports"))
SQLITE_PATH = PROJECT_ROOT / Path(os.getenv("SQLITE_PATH", "neurohypothesis.db"))
LOG_FILE    = PROJECT_ROOT / Path(os.getenv("LOG_FILE",    "logs/neurohypothesis.log"))

# =============================================================================
# Models + determinism (v2.1: seed pinned, temperatures lowered)
# =============================================================================

# =============================================================================
# Models
# =============================================================================
MAIN_LLM_MODEL       = "gpt-4o-mini"
JUDGE_LLM_MODEL      = "gpt-4o-mini"
HYPOTHESIS_LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL      = "text-embedding-3-small"
EMBEDDING_DIMENSIONS  = 1536

# Global seed applied to every LLM call (and offset per-hypothesis so H1..H6
# are reproducible but distinct).  text-embedding-3-small does not accept a
# seed parameter — embeddings are deterministic by design.
LLM_SEED = 42

# LLM temperatures: 0.0 for all decision-making; Path C integration uses 0.2
# to allow synthesis creativity across two evidence sources.
MAIN_LLM_TEMPERATURE        = 0.0
JUDGE_LLM_TEMPERATURE       = 0.0
GEN_LLM_TEMPERATURE         = 0.0
INTEGRATION_LLM_TEMPERATURE = 0.2

# =============================================================================
# Pricing  (USD per 1 000 000 tokens — verify at platform.openai.com/pricing)
# =============================================================================

PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini":            {"input": 0.15,  "output": 0.60},
    "gpt-4o":                 {"input": 2.50,  "output": 10.00},
    "text-embedding-3-small": {"input": 0.02,  "output": 0.00},
    "text-embedding-3-large": {"input": 0.13,  "output": 0.00},
}

# =============================================================================
# Chunking
# =============================================================================

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80

SECTION_TYPES = ["abstract", "introduction", "discussion", "limitations_future"]

# =============================================================================
# Retrieval
# =============================================================================

SEMANTIC_TOP_K = 10
BM25_TOP_K     = 10
FINAL_TOP_K    = 6
RRF_K          = 60     # Reciprocal Rank Fusion smoothing constant

# Max evidence chunks fed to the past/future summarizers and the generator.
# Shared by generate.py (the [:N] slice) and N9 Path C front-loading so PDF
# evidence reserved at the head is guaranteed to fall inside this window.
SUMMARY_MAX_CHUNKS = 12

# ── Path C (combined) — uploaded-PDF evidence merge (N9) ──────────────────────
# How uploaded PDFs are blended with the PubMed per-category evidence.
PATH_C_PDF_MAX_CHUNKS        = 3     # cap on extra PDF *RAG grounding* chunks per
                                     # tag (past/future); metadata signal is uncapped
                                     # so every kept PDF is always represented.
PATH_C_PDF_MIN_QUERY_COSINE  = 0.15  # soft gate: drop a PDF from the combined merge
                                     # only if cosine(query, PDF) is below this; the
                                     # UI then warns it was not used. Path A (PDF-only)
                                     # is never gated — there the PDF is the sole source.
PATH_C_PDF_CATEGORY_BOOST    = 0.10  # additive rank bonus when a PDF's category is in
                                     # the current hypothesis's primary+complementary
                                     # pair (ordering only; never excludes).

# =============================================================================
# PubMed E-utilities (v2.1: 25-year window, per-category top-10)
# =============================================================================

PUBMED_BASE_URL          = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_PER_CATEGORY_N    = 10      # papers fetched per category in Path B
PUBMED_YEARS_BACK        = 25      # primary search window (was 5 in v2)
PUBMED_FALLBACK_NO_YEAR  = True    # if primary search returns 0, retry without year filter
PUBMED_MAX_RETRIES       = 3       # network retries (with exponential backoff)
PUBMED_TIMEOUT_S         = 15      # seconds per request
PUBMED_EMAIL             = os.getenv("PUBMED_EMAIL", "")  # required by NCBI fair-use policy — set in .env or Streamlit secrets

# PubMed external-novelty check (shared by Paths A, B, C — runs per hypothesis)
PUBMED_CHECK_YEARS_BACK      = 25     # prior-art window (was 5; aligns with evidence retrieval)
PUBMED_CHECK_TOP_N           = 20     # candidate pool fetched, then cosine-reranked (was 5)
PUBMED_CHECK_MATCH_THRESHOLD = 0.55   # cosine >= this counts as a real match (calibrate empirically)

# Predatory / low-quality publisher denylist.
# Case-insensitive substring match against journal title.
# v2.1: no publication-type filter (Reviews / Editorials / Letters all stay).
PREDATORY_PUBLISHERS: frozenset[str] = frozenset({
    "mdpi",
    "hindawi",
})

# =============================================================================
# Per-category retrieval loop (v2.1, new — Path B core)
# =============================================================================

# Temperature escalation across reformulation attempts.
# Attempt 1 starts at 0.0; if quality-check fails or post-retrieval relevance
# is below threshold, attempt 2 uses 0.2, then attempt 3 uses 0.4.
REFORMULATE_TEMP_ESCALATION = [0.0, 0.0]   # deterministic — context feedback drives variation
REFORMULATE_MAX_ATTEMPTS    = 2

# Per-abstract relevance threshold: cosine similarity of each abstract to the
# original user query.  Note: complement categories (Genetics, Computational,
# etc.) will always score lower than the primary category against queries that
# mention specific brain regions — 0.22 ensures they aren't unfairly excluded.
RELEVANCE_THRESHOLD_PER_ABSTRACT = 0.22

# Minimum number of papers that must pass the per-abstract relevance check.
# 3 is sufficient for cross-scale hypothesis generation from complement categories.
MIN_RELEVANT_ABSTRACTS = 3

# =============================================================================
# Session / PDF limits
# =============================================================================

MAX_PDF_COUNT      = 3          # hard cap on user-uploaded PDFs
MAX_PDF_SIZE_MB    = 20         # per-file size cap
MAX_QUERY_LENGTH   = 1000       # characters
MIN_QUERY_LENGTH   = 5
MAX_HYPOTHESES     = 6          # hard cap per session (v2.1: was 3)

# Path A (PDF-only): incremental 1->N loop, gap-pair rotation, diversity gate
PATH_A_MAX_HYPOTHESES  = 6      # user can Continue up to this many (one at a time)
PATH_A_GEN_TEMPERATURE = 0.3    # diversity across the batch
PATH_A_SUMMARY_BULLETS = 5      # past/future bullets (was 3) -> 5x5 pair grid

# Gap-pair selection (Path A): a good pair is RELEVANT yet DIVERGENT.
GAP_BAND_LOW   = 0.35   # below: past/future unrelated (not a meaningful gap)
GAP_BAND_HIGH  = 0.70   # above: future just restates past (no gap)
GAP_MMR_LAMBDA = 0.7    # MMR weight: gap quality vs diversity-from-already-picked
DIVERSITY_GATE_THRESHOLD = 0.75  # regenerate if a new hyp is >= this cosine to a prior one

# =============================================================================
# Quality gate (Decision D — N16)
# =============================================================================

ORIGINALITY_PASS_THRESHOLD = 0.2   # originality_score must be >= this to pass
PLAUSIBILITY_PASS_AVG      = 3.0   # plausibility_avg must be >= this to pass
QUALITY_GATE_MAX_ATTEMPTS  = 3     # regeneration attempts before showing best-of-3

# =============================================================================
# Originality grading thresholds (shared by originality and gap scoring)
#   similarity <= VERY_ORIGINAL_THRESHOLD              → "Very original"
#   VERY < sim < LESS                                   → "Moderately original"
#   similarity >= LESS_ORIGINAL_THRESHOLD              → "Less original"
# =============================================================================

VERY_ORIGINAL_THRESHOLD = 0.3
LESS_ORIGINAL_THRESHOLD = 0.8

BLUE_GRADE_COLORS: dict[str, str] = {
    "very":     "#5e3a87",   # dark purple   (matches button)
    "moderate": "#8a6cb4",   # medium purple (matches score badges)
    "less":     "#b8a0d4",   # light purple
}
ORIGINALITY_COLORS = BLUE_GRADE_COLORS
GAP_COLORS         = BLUE_GRADE_COLORS

# =============================================================================
# Temporal tagging (past vs future chunk classification)
# =============================================================================

TEMPORAL_REFERENCES: dict[str, str] = {
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

TEMPORAL_NEUTRAL_MARGIN = 0.1   # if |past_sim - future_sim| < this → "neutral"

# =============================================================================
# Path routing (v2.1, new)
# =============================================================================

# User-facing path choices (driven by the path selector in the UI).
#   local_only   — Path A: use uploaded PDFs only.
#   pubmed_only  — Path B: PubMed abstracts only.
#   combined     — Path C: integrate both PDF-derived and PubMed-derived hyps.
PATH_OPTIONS = ("local_only", "pubmed_only", "combined")

# =============================================================================
# Categorisation system — 6 categories (v2.1: "Other" removed; redistributed)
# =============================================================================

# Canonical category names.  Used as dict keys throughout the agent.
# Alphabetical order — this is the order Path B uses for complementary
# pairing (after removing the primary category).
CATEGORIES: list[str] = [
    "Animal Models",
    "Behavioral & Cognitive Neuroscience",
    "Computational & Theoretical",
    "Genetics & Molecular Biology",
    "Human Neuroimaging",
    "Postmortem & Ex-Vivo Histology",
]

# Human-readable descriptions fed to the LLM primary-category picker and
# per-category reformulator.  Each description includes representative
# MeSH-aligned terminology so the LLM can match query meaning to category.
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Animal Models": (
        "IN-VIVO preclinical and basic research in animals. "
        "Includes Mice, Rats, Macaca, Zebrafish, C. elegans, Drosophila, "
        "Disease Models Animal, in-vivo electrophysiology, optogenetics, "
        "behavioural testing in animals, neuropharmacology, "
        "neurochemistry, neuroendocrinology, neurogenesis, neural plasticity, "
        "and developmental neuroscience conducted in living animals. "
        "Note: ex-vivo tissue work in animals (e.g. slice electrophysiology, "
        "in-vitro techniques) is NOT part of this category — "
        "it belongs to 'Postmortem & Ex-Vivo Histology'."
    ),
    "Behavioral & Cognitive Neuroscience": (
        "Behavioural, cognitive, affective, and social studies in humans "
        "without primary imaging or genetic readouts.  Includes "
        "psychophysics, neuropsychology, behavioural tasks, "
        "neuropsychological tests, cognitive testing batteries, lesion "
        "studies, emotion/reward/stress/empathy paradigms, social cognition, "
        "and clinical neuroscience studies in humans where the primary "
        "readout is behavioural, cognitive, or clinical-outcome. "
        "Also includes clinical trials, drug "
        "repurposing, case reports, screening-tool development, "
        "epidemiology, and population-level risk-factor / cohort studies."
    ),
    "Computational & Theoretical": (
        "Modelling, simulation, and theoretical work.  Computational "
        "neuroscience, biophysical models (Hodgkin-Huxley, neural mass), "
        "connectome modelling, machine learning/AI applied to brain data, "
        "dynamical systems, and theoretical frameworks. "
        "Also includes brain-computer interfaces "
        "(BCIs), neuroprostheses, DBS devices, neurofeedback systems, "
        "neuroengineering and neurotechnology papers, statistical methods, "
        "software/toolbox releases, and signal-processing algorithms."
    ),
    "Genetics & Molecular Biology": (
        "Genetic, transcriptomic, proteomic, and molecular work in any "
        "species.  Includes GWAS, knock-out/knock-in models, transgenic "
        "animals, single-cell/single-nucleus RNA-seq, transcriptomics, "
        "proteomics, epigenetics, in situ hybridization, CRISPR, and gene "
        "expression studies.  Also includes molecular & cellular "
        "neuroscience (synaptic biology, signal transduction, ion channels) "
        "and neurogenetics/neuroepigenetics.  Neuroinflammation, microglia, "
        "and glial / vascular / metabolic neuroscience when the work is "
        "molecular."
    ),
    "Human Neuroimaging": (
        "Non-invasive imaging and recording in LIVING humans.  Methods "
        "include structural and functional MRI, diffusion MRI, MR "
        "spectroscopy, EEG, MEG, PET, SPECT, fNIRS, TMS, tDCS, and "
        "ultra-high-field (7T+) MRI."
    ),
    "Postmortem & Ex-Vivo Histology": (
        "Postmortem and ex-vivo tissue work in ANY species.  Includes "
        "postmortem human/primate brain neuropathology, immunohistochemistry, "
        "stereology, postmortem MRI ex vivo, postmortem proteomics/"
        "transcriptomics — AND ALSO animal ex-vivo work: brain slice "
        "preparations, in-vitro techniques, ex-vivo electrophysiology, "
        "tissue culture-based optogenetics. "
        "Ex-vivo and in-vitro tissue work in animals also belongs here "
        "(the distinguishing principle is live-tissue-removed-from-body, "
        "regardless of species). "
        "Also includes STEM CELL and ORGANOID neuroscience: iPSC-derived "
        "neurons/glia, cerebral organoids, brain organoids, neural organoids, "
        "cortical organoids, assembloids, and any 3D in-vitro brain tissue "
        "model regardless of whether derived from human or animal cells."
    ),
}

# Audit table: how 17 subfields map to the 6 categories.
# Used for traceability and reviewer explanation — not used in runtime logic.
# v2.1: "Other"-mapped subfields redistributed.
SUBFIELD_TO_CATEGORY: dict[str, str] = {
    "Molecular & Cellular Neuroscience":              "Genetics & Molecular Biology",
    "Developmental Neuroscience":                     "Animal Models",
    "Systems Neuroscience":                           "Animal Models",
    "Cognitive Neuroscience":                         "Behavioral & Cognitive Neuroscience",
    "Behavioral Neuroscience":                        "Animal Models",
    "Affective & Social Neuroscience":                "Behavioral & Cognitive Neuroscience",
    "Computational & Theoretical Neuroscience":       "Computational & Theoretical",
    "Clinical & Translational Neuroscience (human)":  "Behavioral & Cognitive Neuroscience",
    "Clinical & Translational Neuroscience (animal)": "Animal Models",
    "Neuroimaging & Human Systems Methods":           "Human Neuroimaging",
    "Neurophysiology & Electrophysiology":            "Animal Models",
    "Neurogenetics & Neuroepigenetics":               "Genetics & Molecular Biology",
    "Neuropharmacology & Neurochemistry":             "Animal Models",
    "Neuroendocrinology & Homeostatic Neuroscience":  "Animal Models",
    "Comparative, Evolutionary & Ethological Neuroscience": "Animal Models",
    # Former "Other" — v2.1 redistribution:
    "Neuroengineering & Neurotechnology":             "Computational & Theoretical",
    "Neuroimmunology & Neuroinflammation":            "Animal Models",   # also valid in Genetics
    "Glial, Neurovascular & Metabolic Neuroscience":  "Animal Models",   # also valid in Genetics
}

# ---------------------------------------------------------------------------
# MeSH → Category lookup dict (v2.1: "Other" entries redistributed; 6 cats).
#
# NOTE: in v2.1 the categorizer module is removed.  This MeSH map is no
# longer consulted at runtime for paper bucketing.  It remains here as:
#   (a) reference context for the LLM primary-category picker prompt,
#   (b) hints inside the per-category reformulation prompts, and
#   (c) reviewer / documentation traceability.
#
# Some entries appear in two categories where appropriate (neuroinflammation,
# microglia, glial-vascular-metabolic items map to both Animal Models and
# Genetics — disambiguation happens by the LLM at primary-pick time, not
# by a deterministic categorizer).
# ---------------------------------------------------------------------------

MESH_TO_CATEGORY: dict[str, str] = {

    # ── Human Neuroimaging ────────────────────────────────────────────────
    "Magnetic Resonance Imaging":              "Human Neuroimaging",
    "Functional Magnetic Resonance Imaging":   "Human Neuroimaging",
    "Diffusion Magnetic Resonance Imaging":    "Human Neuroimaging",
    "Magnetic Resonance Spectroscopy":         "Human Neuroimaging",
    "Electroencephalography":                  "Human Neuroimaging",
    "Magnetoencephalography":                  "Human Neuroimaging",
    "Positron-Emission Tomography":            "Human Neuroimaging",
    "Tomography, Emission-Computed, Single-Photon": "Human Neuroimaging",
    "Spectroscopy, Near-Infrared":             "Human Neuroimaging",
    "Transcranial Magnetic Stimulation":       "Human Neuroimaging",
    "Transcranial Direct Current Stimulation": "Human Neuroimaging",
    "Neuroimaging":                            "Human Neuroimaging",
    "Functional Neuroimaging":                 "Human Neuroimaging",
    "Brain Mapping":                           "Human Neuroimaging",
    # EEG rhythms and evoked potentials
    "Evoked Potentials":                       "Human Neuroimaging",
    "Evoked Potentials, Visual":               "Human Neuroimaging",
    "Evoked Potentials, Auditory":             "Human Neuroimaging",
    "Evoked Potentials, Motor":                "Human Neuroimaging",
    "Brain Waves":                             "Human Neuroimaging",
    "Alpha Rhythm":                            "Human Neuroimaging",
    "Beta Rhythm":                             "Human Neuroimaging",
    "Theta Rhythm":                            "Human Neuroimaging",
    "Gamma Rhythm":                            "Human Neuroimaging",
    # Ultrasound / other modalities
    "Ultrasonography, Doppler, Transcranial":  "Human Neuroimaging",
    "Cerebrovascular Circulation":             "Human Neuroimaging",
    "Cerebral Blood Flow":                     "Human Neuroimaging",
    "Electrocorticography":                    "Human Neuroimaging",
    "Multimodal Imaging":                      "Human Neuroimaging",
    "Disease Models, Animal":                  "Animal Models",
    "Mice":                                    "Animal Models",
    "Rats":                                    "Animal Models",
    "Macaca":                                  "Animal Models",
    "Callithrix":                              "Animal Models",
    "Primates":                                "Animal Models",
    "Haplorhini":                              "Animal Models",
    "Cats":                                    "Animal Models",
    "Rabbits":                                 "Animal Models",
    "Swine":                                   "Animal Models",
    "Ferrets":                                 "Animal Models",
    "Zebrafish":                               "Animal Models",
    "Caenorhabditis elegans":                  "Animal Models",
    "Drosophila":                              "Animal Models",
    "Animals":                                 "Animal Models",
    "Patch-Clamp Techniques":                  "Postmortem & Ex-Vivo Histology",
    "Electrophysiology":                       "Animal Models",
    "Neurophysiology":                         "Animal Models",
    "Electrophysiological Phenomena":          "Animal Models",
    "Action Potentials":                       "Animal Models",
    "Membrane Potentials":                     "Animal Models",
    "Neural Conduction":                       "Animal Models",
    "Microdialysis":                           "Animal Models",
    "Stereotaxic Techniques":                  "Animal Models",
    "Electrodes, Implanted":                   "Animal Models",
    "Optogenetics":                            "Animal Models",
    "In Vitro Techniques":                     "Postmortem & Ex-Vivo Histology",
    "Behavior, Animal":                        "Animal Models",
    "Conditioning, Operant":                   "Animal Models",
    "Psychophysiology":                        "Animal Models",
    "Ethology":                                "Animal Models",
    "Evolution, Biological":                   "Animal Models",
    "Phylogeny":                               "Animal Models",
    "Anatomy, Comparative":                    "Animal Models",
    "Physiology, Comparative":                 "Animal Models",
    "Species Specificity":                     "Animal Models",
    "Encephalization":                         "Animal Models",
    "Adaptation, Biological":                  "Animal Models",
    "Natural Selection, Genetic":              "Animal Models",
    "Neuropharmacology":                       "Animal Models",
    "Neurochemistry":                          "Animal Models",
    "Neurotransmitter Agents":                 "Animal Models",
    "Receptors, Neurotransmitter":             "Animal Models",
    "Central Nervous System Agents":           "Animal Models",
    "Brain Chemistry":                         "Animal Models",
    "Neuroendocrinology":                      "Animal Models",
    "Hypothalamus":                            "Animal Models",
    "Pituitary Gland":                         "Animal Models",
    "Hypothalamo-Hypophyseal System":          "Animal Models",
    "Hypothalamic Hormones":                   "Animal Models",
    "Autonomic Nervous System":                "Animal Models",
    "Circadian Rhythm":                        "Animal Models",
    "Homeostasis":                             "Animal Models",
    "Cell Differentiation":                    "Animal Models",
    "Embryonic and Fetal Development":         "Animal Models",
    "Nervous System/embryology":               "Animal Models",
    "Neuronal Plasticity":                     "Animal Models",
    "Neural Networks, Physiological":          "Animal Models",
    "Translational Research, Biomedical":      "Animal Models",
    # Disease models
    "Alzheimer Disease":                        "Animal Models",
    "Parkinson Disease":                        "Animal Models",
    "Epilepsy":                                 "Animal Models",
    "Stroke":                                   "Animal Models",
    "Brain Injuries, Traumatic":                "Animal Models",
    # Behavioral assays
    # Circuit tools
    "Channelrhodopsins":                        "Animal Models",
    "Calcium Signaling":                        "Animal Models",
    "Reporter Genes":                           "Animal Models",
    "Neuroinflammation":                       "Animal Models",
    "Energy Metabolism":                       "Animal Models",

    # ── Genetics & Molecular Biology ──────────────────────────────────────
    "Genome-Wide Association Study":           "Genetics & Molecular Biology",
    "Genetics":                                "Genetics & Molecular Biology",
    "Genomics":                                "Genetics & Molecular Biology",
    "Mice, Knockout":                          "Genetics & Molecular Biology",
    "Mice, Transgenic":                        "Genetics & Molecular Biology",
    "Gene Expression Profiling":               "Genetics & Molecular Biology",
    "Transcriptome":                           "Genetics & Molecular Biology",
    "Sequence Analysis, RNA":                  "Genetics & Molecular Biology",
    "Sequence Analysis, DNA":                  "Genetics & Molecular Biology",
    "Exome Sequencing":                        "Genetics & Molecular Biology",
    "Single-Cell Analysis":                    "Genetics & Molecular Biology",
    "Epigenomics":                             "Genetics & Molecular Biology",
    "Epigenesis, Genetic":                     "Genetics & Molecular Biology",
    "DNA Methylation":                         "Genetics & Molecular Biology",
    "Chromatin Immunoprecipitation":           "Genetics & Molecular Biology",
    "CRISPR-Cas Systems":                      "Genetics & Molecular Biology",
    "Gene Knockout Techniques":                "Genetics & Molecular Biology",
    "RNA Interference":                        "Genetics & Molecular Biology",
    "MicroRNAs":                               "Genetics & Molecular Biology",
    "RNA, Long Noncoding":                     "Genetics & Molecular Biology",
    "Polymorphism, Single Nucleotide":         "Genetics & Molecular Biology",
    "Genetic Predisposition to Disease":       "Genetics & Molecular Biology",
    "Genetics, Behavioral":                    "Genetics & Molecular Biology",
    "Mendelian Randomization Analysis":        "Genetics & Molecular Biology",
    "Metabolomics":                            "Genetics & Molecular Biology",
    "Lipidomics":                              "Genetics & Molecular Biology",
    "Nervous System Diseases/genetics":        "Genetics & Molecular Biology",
    "Molecular Biology":                       "Genetics & Molecular Biology",
    "Cell Biology":                            "Genetics & Molecular Biology",
    "Neurobiology":                            "Genetics & Molecular Biology",
    "Synaptic Transmission":                   "Genetics & Molecular Biology",
    "Signal Transduction":                     "Genetics & Molecular Biology",
    "Apolipoproteins E":                       "Genetics & Molecular Biology",
    # New additions
    "Real-Time Polymerase Chain Reaction":     "Genetics & Molecular Biology",
    "Mass Spectrometry":                       "Genetics & Molecular Biology",
    "RNA, Small Interfering":                  "Genetics & Molecular Biology",
    "Antisense Oligonucleotides":              "Genetics & Molecular Biology",
    "Lentivirus":                              "Genetics & Molecular Biology",
    "Genetic Vectors":                         "Genetics & Molecular Biology",
    "Neurofilament Proteins":                  "Genetics & Molecular Biology",
    "Cerebrospinal Fluid":                     "Genetics & Molecular Biology",

    # ── Behavioral & Cognitive Neuroscience ───────────────────────────────
    "Cognition":                               "Behavioral & Cognitive Neuroscience",
    "Memory":                                  "Behavioral & Cognitive Neuroscience",
    "Attention":                               "Behavioral & Cognitive Neuroscience",
    "Executive Function":                      "Behavioral & Cognitive Neuroscience",
    "Neuropsychological Tests":                "Behavioral & Cognitive Neuroscience",
    "Psychophysics":                           "Behavioral & Cognitive Neuroscience",
    "Reaction Time":                           "Behavioral & Cognitive Neuroscience",
    "Cognitive Dysfunction":                   "Behavioral & Cognitive Neuroscience",
    "Cognitive Aging":                         "Behavioral & Cognitive Neuroscience",
    "Learning":                                "Behavioral & Cognitive Neuroscience",
    "Behavior":                                "Behavioral & Cognitive Neuroscience",
    "Behavioral Research":                     "Behavioral & Cognitive Neuroscience",
    "Behavior Rating Scale":                   "Behavioral & Cognitive Neuroscience",
    "Psychology":                              "Behavioral & Cognitive Neuroscience",
    "Emotions":                                "Behavioral & Cognitive Neuroscience",
    "Affect":                                  "Behavioral & Cognitive Neuroscience",
    "Social Behavior":                         "Behavioral & Cognitive Neuroscience",
    "Empathy":                                 "Behavioral & Cognitive Neuroscience",
    "Fear":                                    "Behavioral & Cognitive Neuroscience",
    "Anxiety":                                 "Behavioral & Cognitive Neuroscience",
    "Reward":                                  "Behavioral & Cognitive Neuroscience",
    "Stress, Psychological":                   "Behavioral & Cognitive Neuroscience",
    "Quality of Life":                         "Behavioral & Cognitive Neuroscience",
    "Activities of Daily Living":              "Behavioral & Cognitive Neuroscience",
    "Pain":                                    "Behavioral & Cognitive Neuroscience",
    "Inhibition, Psychological":               "Behavioral & Cognitive Neuroscience",
    "Working Memory":                          "Behavioral & Cognitive Neuroscience",
    "Clinical Trials as Topic":                "Behavioral & Cognitive Neuroscience",
    "Therapeutics":                            "Behavioral & Cognitive Neuroscience",
    "Neurology":                               "Behavioral & Cognitive Neuroscience",
    "Nervous System Diseases":                 "Behavioral & Cognitive Neuroscience",
    "Central Nervous System Diseases":         "Behavioral & Cognitive Neuroscience",
    "Biomarkers":                              "Behavioral & Cognitive Neuroscience",
    "Drug Repositioning":                      "Behavioral & Cognitive Neuroscience",
    "Drug Repurposing":                        "Behavioral & Cognitive Neuroscience",
    "Epidemiologic Studies":                   "Behavioral & Cognitive Neuroscience",
    "Cohort Studies":                          "Behavioral & Cognitive Neuroscience",
    "Risk Factors":                            "Behavioral & Cognitive Neuroscience",
    "Population Surveillance":                 "Behavioral & Cognitive Neuroscience",
    "Mass Screening":                          "Behavioral & Cognitive Neuroscience",
    "Surveys and Questionnaires":              "Behavioral & Cognitive Neuroscience",
    "Case Reports":                            "Behavioral & Cognitive Neuroscience",
    # Behavioral assays
    "Exploratory Behavior":                    "Behavioral & Cognitive Neuroscience",
    "Open Field Test":                         "Behavioral & Cognitive Neuroscience",
    "Motor Activity":                          "Behavioral & Cognitive Neuroscience",
    "Locomotion":                              "Behavioral & Cognitive Neuroscience",
    "Gait":                                    "Behavioral & Cognitive Neuroscience",
    "Maze Learning":                           "Behavioral & Cognitive Neuroscience",
    "Spatial Learning":                        "Behavioral & Cognitive Neuroscience",
    "Avoidance Learning":                      "Behavioral & Cognitive Neuroscience",
    "Conditioning, Classical":                 "Behavioral & Cognitive Neuroscience",
    "Extinction, Psychological":               "Behavioral & Cognitive Neuroscience",
    "Recognition, Psychology":                 "Behavioral & Cognitive Neuroscience",
    "Aggression":                              "Behavioral & Cognitive Neuroscience",
    "Anhedonia":                               "Behavioral & Cognitive Neuroscience",
    "Prepulse Inhibition":                     "Behavioral & Cognitive Neuroscience",
    "Startle Reaction":                        "Behavioral & Cognitive Neuroscience",
    "Self Administration":                     "Behavioral & Cognitive Neuroscience",
    "Delay Discounting":                       "Behavioral & Cognitive Neuroscience",
    # Affective / clinical
    "Depressive Disorder":                     "Behavioral & Cognitive Neuroscience",
    "Anxiety Disorders":                       "Behavioral & Cognitive Neuroscience",
    "Schizophrenia":                           "Behavioral & Cognitive Neuroscience",
    "Autism Spectrum Disorder":                "Behavioral & Cognitive Neuroscience",
    "Attention Deficit Disorder with Hyperactivity": "Behavioral & Cognitive Neuroscience",
    # Psychophysical / assessment
    "Intelligence Tests":                      "Behavioral & Cognitive Neuroscience",
    "Psychomotor Performance":                 "Behavioral & Cognitive Neuroscience",
    "Eye Movements":                           "Behavioral & Cognitive Neuroscience",
    "Video Recording":                         "Behavioral & Cognitive Neuroscience",

    # ── Computational & Theoretical ───────────────────────────────────────
    "Computer Simulation":                     "Computational & Theoretical",
    "Models, Neurological":                    "Computational & Theoretical",
    "Models, Theoretical":                     "Computational & Theoretical",
    "Computational Biology":                   "Computational & Theoretical",
    "Neural Networks, Computer":               "Computational & Theoretical",
    "Deep Learning":                           "Computational & Theoretical",
    "Machine Learning":                        "Computational & Theoretical",
    "Artificial Intelligence":                 "Computational & Theoretical",
    "Algorithms":                              "Computational & Theoretical",
    "Nonlinear Dynamics":                      "Computational & Theoretical",
    "Bayes Theorem":                           "Computational & Theoretical",
    "Connectome":                              "Computational & Theoretical",
    "Stereology":                              "Computational & Theoretical",
    "Reinforcement, Psychology":               "Computational & Theoretical",
    "Systems Biology":                         "Computational & Theoretical",
    "Bioinformatics":                          "Computational & Theoretical",
    "Information Theory":                      "Computational & Theoretical",
    "Data Mining":                             "Computational & Theoretical",
    "Regression Analysis":                     "Computational & Theoretical",
    "Markov Chains":                           "Computational & Theoretical",
    "Principal Component Analysis":            "Computational & Theoretical",
    "Brain-Computer Interfaces":               "Computational & Theoretical",
    "Neurofeedback":                           "Computational & Theoretical",
    "Deep Brain Stimulation":                  "Computational & Theoretical",
    "Implantable Neurostimulators":            "Computational & Theoretical",
    "Prostheses and Implants":                 "Computational & Theoretical",
    "Software":                                "Computational & Theoretical",
    "Signal Processing, Computer-Assisted":    "Computational & Theoretical",
    "Image Processing, Computer-Assisted":     "Computational & Theoretical",
    "Statistics as Topic":                     "Computational & Theoretical",
    "Models, Statistical":                     "Computational & Theoretical",
    "Entropy":                                 "Computational & Theoretical",
    "Stochastic Processes":                    "Computational & Theoretical",
    "Decision Making":                         "Computational & Theoretical",

    # ── Postmortem & Ex-Vivo Histology ────────────────────────────────────
    "Autopsy":                                       "Postmortem & Ex-Vivo Histology",
    "Postmortem Changes":                            "Postmortem & Ex-Vivo Histology",
    "Immunohistochemistry":                          "Postmortem & Ex-Vivo Histology",
    "Pathology":                                     "Postmortem & Ex-Vivo Histology",
    "Neuropathology":                                "Postmortem & Ex-Vivo Histology",
    # Tissue preparation
    "Tissue Fixation":                               "Postmortem & Ex-Vivo Histology",
    "Microtomy":                                     "Postmortem & Ex-Vivo Histology",
    "Histological Techniques":                       "Postmortem & Ex-Vivo Histology",
    "Staining and Labeling":                         "Postmortem & Ex-Vivo Histology",
    # Microscopy
    "Microscopy":                                    "Postmortem & Ex-Vivo Histology",
    "Microscopy, Confocal":                          "Postmortem & Ex-Vivo Histology",
    "Microscopy, Fluorescence":                      "Postmortem & Ex-Vivo Histology",
    "Microscopy, Fluorescence, Multiphoton":         "Postmortem & Ex-Vivo Histology",
    "Microscopy, Electron":                          "Postmortem & Ex-Vivo Histology",
    "Microscopy, Electron, Transmission":            "Postmortem & Ex-Vivo Histology",
    "Microscopy, Electron, Scanning":                "Postmortem & Ex-Vivo Histology",
    "Imaging, Three-Dimensional":                    "Postmortem & Ex-Vivo Histology",
    # Molecular detection
    "Immunofluorescence":                            "Postmortem & Ex-Vivo Histology",
    "In Situ Hybridization":                         "Postmortem & Ex-Vivo Histology",
    "In Situ Hybridization, Fluorescence":           "Postmortem & Ex-Vivo Histology",
    # Neuronal morphology
    "Dendrites":                                     "Postmortem & Ex-Vivo Histology",
    "Dendritic Spines":                              "Postmortem & Ex-Vivo Histology",
    "Axons":                                         "Postmortem & Ex-Vivo Histology",
    # Myelin / white matter
    "Myelin Sheath":                                 "Postmortem & Ex-Vivo Histology",
    "Demyelinating Diseases":                        "Postmortem & Ex-Vivo Histology",
    "Oligodendroglia":                               "Postmortem & Ex-Vivo Histology",
    # Glial reactivity
    "Gliosis":                                       "Postmortem & Ex-Vivo Histology",
    # Synapses
    "Synapses":                                      "Postmortem & Ex-Vivo Histology",
    # Neurotransmitter / activity markers
    "Tyrosine Hydroxylase":                          "Postmortem & Ex-Vivo Histology",
    "Dopamine":                                      "Postmortem & Ex-Vivo Histology",
    "gamma-Aminobutyric Acid":                       "Postmortem & Ex-Vivo Histology",
    "Serotonin":                                     "Postmortem & Ex-Vivo Histology",
    "Proto-Oncogene Proteins c-fos":                 "Postmortem & Ex-Vivo Histology",
    # Neurogenesis / proliferation
    "Neurogenesis":                                  "Postmortem & Ex-Vivo Histology",
    "Cell Proliferation":                            "Postmortem & Ex-Vivo Histology",
    # Disease pathology markers
    "Plaque, Amyloid":                               "Postmortem & Ex-Vivo Histology",
    "Amyloid beta-Peptides":                         "Postmortem & Ex-Vivo Histology",
    "Neurofibrillary Tangles":                       "Postmortem & Ex-Vivo Histology",
    "tau Proteins":                                  "Postmortem & Ex-Vivo Histology",
    "alpha-Synuclein":                               "Postmortem & Ex-Vivo Histology",
    "TDP-43":                                        "Postmortem & Ex-Vivo Histology",
    "Prions":                                        "Postmortem & Ex-Vivo Histology",
    "Prion Diseases":                                "Postmortem & Ex-Vivo Histology",
    "Ubiquitin":                                     "Postmortem & Ex-Vivo Histology",
    # Cell death
    "Apoptosis":                                     "Postmortem & Ex-Vivo Histology",
    "Cell Death":                                    "Postmortem & Ex-Vivo Histology",
    # Connectivity / tract tracing
    "Neuroanatomical Tract-Tracing Techniques":      "Postmortem & Ex-Vivo Histology",
    "Neural Pathways":                               "Postmortem & Ex-Vivo Histology",
    # Vasculature / extracellular matrix
    "Blood-Brain Barrier":                           "Postmortem & Ex-Vivo Histology",
    "Extracellular Matrix":                          "Postmortem & Ex-Vivo Histology",
    # Structural pathology / atrophy
    "Atrophy":                                       "Postmortem & Ex-Vivo Histology",
    "Brain Injuries":                                "Postmortem & Ex-Vivo Histology",
    "Brain Ischemia":                                "Postmortem & Ex-Vivo Histology",
    # Brain tissue regions
    "Gray Matter":                                   "Postmortem & Ex-Vivo Histology",
    "White Matter":                                  "Postmortem & Ex-Vivo Histology",
    "Cerebral Cortex":                               "Postmortem & Ex-Vivo Histology",
    "Hippocampus":                                   "Postmortem & Ex-Vivo Histology",
    "Brain":                                         "Postmortem & Ex-Vivo Histology",
    "Neurons":                                       "Postmortem & Ex-Vivo Histology",
    "Neuroglia":                                     "Postmortem & Ex-Vivo Histology",
    "Microglia":                                     "Postmortem & Ex-Vivo Histology",
    "Astrocytes":                                    "Postmortem & Ex-Vivo Histology",
    "Brain Diseases":                                "Postmortem & Ex-Vivo Histology",
    "Neurodegenerative Diseases":                    "Postmortem & Ex-Vivo Histology",
    # Non-human primate postmortem (also in Animal Models for in-vivo use)
    "Macaca mulatta":                                "Postmortem & Ex-Vivo Histology",
    "Macaca fascicularis":                           "Postmortem & Ex-Vivo Histology",
    "Pan troglodytes":                               "Postmortem & Ex-Vivo Histology",
    # Vascular + cardiovascular bridging
    "Cerebrovascular Disorders":                     "Postmortem & Ex-Vivo Histology",
    "Cerebral Small Vessel Diseases":                "Postmortem & Ex-Vivo Histology",
    "Arteriosclerosis, Intracranial":                "Postmortem & Ex-Vivo Histology",
    "Brain Infarction":                              "Postmortem & Ex-Vivo Histology",
    "Leukoaraiosis":                                 "Postmortem & Ex-Vivo Histology",
    # Molecular assays on tissue
    "Blotting, Western":                             "Postmortem & Ex-Vivo Histology",
    "Flow Cytometry":                                "Postmortem & Ex-Vivo Histology",
    "Enzyme-Linked Immunosorbent Assay":             "Postmortem & Ex-Vivo Histology",
    "Laser Capture Microdissection":                 "Postmortem & Ex-Vivo Histology",
    "Proteomics":                                    "Postmortem & Ex-Vivo Histology",
    # Organoids / iPSC
    "Organoids":                                     "Postmortem & Ex-Vivo Histology",
    "Brain Organoids":                               "Postmortem & Ex-Vivo Histology",
    "Induced Pluripotent Stem Cells":                "Postmortem & Ex-Vivo Histology",
    "Neural Stem Cells":                             "Postmortem & Ex-Vivo Histology",
    "Stem Cell Research":                            "Postmortem & Ex-Vivo Histology",
    "Stem Cells":                                    "Postmortem & Ex-Vivo Histology",
    "Cell Culture Techniques":                       "Postmortem & Ex-Vivo Histology",
}


# =============================================================================
# Category method keywords for PubMed fallback tiers
# =============================================================================
CATEGORY_METHOD_KEYWORDS: dict[str, str] = {
    "Animal Models": (
        # Species
        "rodent OR mouse OR rat OR zebrafish OR Drosophila OR C. elegans "
        "OR non-human primate OR NHP OR macaque OR marmoset OR monkey OR primate "
        # Genetic models
        "OR transgenic OR knock-out OR knock-in OR Cre-lox OR reporter line "
        "OR conditional knockout OR disease mutation "
        # Lesion / pharmacological models
        "OR excitotoxic lesion OR lesion model OR 6-OHDA OR MPTP "
        "OR surgical lesion OR pharmacological model OR receptor agonist OR receptor antagonist "
        # Behavioral assays
        "OR behavioral assay OR fear conditioning OR open field OR Morris water maze "
        "OR novel object OR social behavior OR locomotion OR anxiety model OR learning memory "
        # Electrophysiology
        "OR single unit recording OR local field potential OR LFP OR patch-clamp "
        "OR in vivo electrophysiology "
        # Circuit tools
        "OR optogenetics OR channelrhodopsin OR chemogenetics OR DREADD "
        "OR calcium imaging OR GCaMP OR miniscope OR two-photon imaging "
        # Viral tracing
        "OR viral tracing OR AAV OR rabies virus OR retrograde tracing OR anterograde tracing "
        # Comparative / evolutionary neuroscience
        "OR comparative neuroscience OR evolutionary neuroscience OR brain evolution "
        "OR encephalization OR phylogenetic OR homology OR cross-species "
        # Disease models
        "OR Alzheimer model OR Parkinson model OR epilepsy model OR stroke model "
        "OR autism model OR TBI OR traumatic brain injury model"
    ),
    "Behavioral & Cognitive Neuroscience": (
        # Locomotion / anxiety / motor assays
        "open field OR elevated plus maze OR light-dark box OR locomotion "
        "OR anxiety-like behavior OR exploration "
        "OR forced swim test OR tail suspension test OR sucrose preference "
        "OR anhedonia OR depression-like behavior "
        "OR rotarod OR balance beam OR grip strength OR gait analysis "
        "OR motor coordination OR motor learning "
        # Spatial / recognition memory
        "OR Morris water maze OR Barnes maze OR radial arm maze "
        "OR spatial learning OR spatial memory "
        "OR novel object recognition OR recognition memory OR episodic memory "
        # Associative / operant learning
        "OR fear conditioning OR associative learning OR extinction "
        "OR operant conditioning OR reinforcement OR Pavlovian conditioning "
        # Social / psychiatric assays
        "OR social interaction OR three-chamber test OR resident-intruder "
        "OR sociability OR social memory OR aggression "
        "OR prepulse inhibition OR startle response OR sensorimotor gating "
        "OR self-administration OR conditioned place preference OR drug-seeking "
        # Cognitive tasks
        "OR Stroop OR Flanker OR Go/No-Go OR stop-signal OR cognitive control "
        "OR N-back OR working memory task OR delay matching-to-sample "
        "OR Wisconsin Card Sorting OR reversal learning OR cognitive flexibility "
        "OR Iowa Gambling Task OR delay discounting OR impulsivity "
        "OR sustained attention OR Posner cueing OR visual search "
        "OR theory of mind OR emotion recognition OR social cognition "
        # Psychophysics
        "OR reaction time OR psychophysics OR signal detection theory "
        "OR eye tracking OR pupilometry "
        # Neuropsychological testing
        "OR MMSE OR MoCA OR WAIS OR CANTAB OR Trail Making Test "
        "OR verbal fluency OR Boston Naming Test OR cognitive assessment "
        "OR neuropsychological testing OR neuropsychological battery "
        # Affective / stress
        "OR Trier stress test OR cortisol OR HPA axis "
        "OR emotion induction OR stress reactivity OR affect "
        # Computational behavior analysis
        "OR drift-diffusion model OR reinforcement-learning model "
        "OR evidence accumulation OR value learning OR prediction error "
        "OR Bayesian behavioral model OR hierarchical model "
        # Behavioral tracking / scoring
        "OR DeepLabCut OR SLEAP OR EthoVision OR pose estimation "
        "OR behavioral tracking OR video tracking OR movement kinematics "
        # Clinical assessment
        "OR ecological momentary assessment OR symptom scale "
        "OR structured interview OR psychiatric assessment OR cognitive OR behavioral"
    ),
    "Computational & Theoretical": (
        # Core modeling
        "computational model OR neural simulation OR circuit model OR biophysical model "
        "OR Hodgkin-Huxley OR integrate-and-fire OR compartmental model "
        # Network models
        "OR recurrent network OR attractor network OR oscillation OR synchronization "
        "OR neural oscillation OR network dynamics "
        # Neural coding
        "OR rate coding OR temporal coding OR population coding OR neural decoding "
        "OR spike train OR neural coding "
        # Machine learning / deep learning
        "OR machine learning OR deep learning OR convolutional neural network "
        "OR recurrent neural network OR transformer OR artificial neural network "
        "OR representation learning "
        # Bayesian / reinforcement learning
        "OR Bayesian inference OR predictive coding OR uncertainty quantification "
        "OR reinforcement learning OR value function OR policy gradient "
        # Dynamical systems
        "OR dynamical systems OR phase space OR bifurcation OR chaos "
        "OR neural trajectory OR attractor dynamics "
        # Connectomics / graph theory
        "OR graph theory OR network topology OR modularity OR hub detection "
        "OR centrality OR connectome analysis "
        # Information theory / stats
        "OR entropy OR mutual information OR information theory "
        "OR GLM OR mixed model OR causal inference OR drift diffusion model "
        # Systems / bioinformatics
        "OR bioinformatics OR systems biology OR functional connectivity OR data-driven"
    ),
    "Genetics & Molecular Biology": (
        # Expression analysis
        "gene expression OR qPCR OR RNA-seq OR bulk RNA-seq "
        "OR single-cell RNA-seq OR scRNA-seq OR snRNA-seq "
        # Epigenetics
        "OR DNA methylation OR ATAC-seq OR ChIP-seq OR histone modification "
        "OR epigenetics OR epigenomics "
        # Genome editing / transgenic
        "OR CRISPR OR CRISPR-Cas9 OR base editing OR prime editing "
        "OR knock-in OR knock-out OR Cre-lox "
        # Viral delivery
        "OR AAV OR lentivirus OR retrovirus OR viral vector OR gene delivery "
        # Proteomics
        "OR proteomics OR mass spectrometry OR phosphoproteomics OR synaptomics "
        # Molecular assays
        "OR Western blot OR ELISA OR immunofluorescence "
        # RNA / spatial detection
        "OR ISH OR FISH OR RNAscope OR in situ hybridization "
        "OR spatial transcriptomics OR Visium OR MERFISH "
        # Single-cell / spatial omics
        "OR single-cell omics OR scATAC-seq OR cell-type profiling OR spatial omics "
        # iPSC / organoids / culture
        "OR iPSC OR organoid OR human neurons OR patient-specific OR primary culture "
        # Molecular perturbation
        "OR RNAi OR shRNA OR siRNA OR antisense oligonucleotide OR overexpression "
        # Biomarkers
        "OR biomarker OR CSF OR cerebrospinal fluid OR neurofilament light "
        "OR plasma amyloid OR tau biomarker OR genetic OR molecular OR GWAS"
    ),
    "Human Neuroimaging": (
        # Structural MRI
        "structural MRI OR cortical thickness OR gray matter volume OR brain morphology "
        "OR voxel-based morphometry OR surface-based morphometry "
        # Functional MRI
        "OR fMRI OR BOLD OR task fMRI OR resting-state fMRI OR resting state "
        "OR default mode network OR functional connectivity "
        # Diffusion MRI
        "OR diffusion MRI OR DTI OR DWI OR tractography OR white matter integrity "
        "OR fractional anisotropy OR FA OR MD "
        # PET / SPECT
        "OR PET OR amyloid PET OR tau PET OR dopamine PET OR receptor binding "
        "OR neuroinflammation PET OR SPECT OR perfusion imaging "
        # EEG / MEG
        "OR EEG OR MEG OR electroencephalography OR magnetoencephalography "
        "OR oscillation OR ERP OR evoked potential OR source localization "
        "OR brain waves OR alpha rhythm OR beta rhythm "
        # fNIRS / MRS
        "OR fNIRS OR NIRS OR cortical oxygenation OR hemodynamics "
        "OR MRS OR spectroscopy OR GABA OR glutamate OR NAA "
        # Stimulation
        "OR TMS OR transcranial magnetic stimulation OR motor mapping "
        "OR cortical excitability OR tDCS OR tACS OR oscillatory entrainment "
        "OR focused ultrasound OR FUS OR brain stimulation "
        # Intracranial / invasive
        "OR intracranial EEG OR ECoG OR stereo-EEG OR SEEG OR electrocorticography "
        # Connectivity / multimodal
        "OR structural connectivity OR network analysis OR multimodal imaging "
        "OR MRI-PET OR EEG-fMRI OR imaging-genetics "
        # Ultra-high field
        "OR ultra-high field MRI OR 7T MRI OR 7 Tesla OR UHF MRI"
    ),
    "Postmortem & Ex-Vivo Histology": (
        # Core method identifiers
        "postmortem OR histology OR neuropathology OR brain tissue OR ex vivo "
        "OR autopsy OR immunohistochemistry OR histological "
        # General stains
        "OR nissl stain OR hematoxylin OR H&E OR golgi stain "
        "OR luxol fast blue OR myelin stain "
        # Microscopy
        "OR microscopy OR confocal microscopy OR electron microscopy "
        "OR light sheet microscopy OR two-photon microscopy "
        "OR fluorescence microscopy OR stereology "
        # 3D / tissue clearing
        "OR tissue clearing OR CLARITY OR iDISCO OR CUBIC "
        # IHC / IF cell-type markers
        "OR GFAP OR NeuN OR Iba1 OR MAP2 OR neurofilament "
        "OR synaptophysin OR PSD-95 "
        # RNA detection
        "OR RNAscope OR FISH OR ISH OR spatial transcriptomics "
        # Connectivity / tracing
        "OR tract tracing OR neural tracer OR AAV OR viral tracing "
        # Pathology markers
        "OR amyloid OR amyloid-beta OR tau OR tauopathy OR p-tau "
        "OR alpha-synuclein OR TDP-43 OR prion OR ubiquitin "
        # Structural / vascular
        "OR atrophy OR brain lesion OR neurodegeneration OR demyelination "
        "OR blood-brain barrier OR vascular integrity "
        # Cell death
        "OR apoptosis OR cell death OR TUNEL "
        # Molecular assays on tissue
        "OR western blot OR ELISA OR flow cytometry "
        # NHP postmortem
        "OR macaque brain OR chimpanzee brain OR primate postmortem"
    ),
}

# =============================================================================
# UI strings
# =============================================================================

APP_TITLE    = "Neurohypothesis"
APP_SUBTITLE = (
    "Generates cross-scale neuroscience hypotheses from your uploaded papers "
    "and live PubMed search."
)
DISCLAIMER = (
    "AI-generated hypotheses are research suggestions, not scientific facts. "
    "All outputs require expert validation before use in actual research."
)
