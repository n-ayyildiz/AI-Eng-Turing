# Build Steps — Hypothesis Generator

This document records what was built at each stage of development.

---

## Step (a) — Scaffolding

- Folder structure created: `src/`, `data/papers/`, `data/eval_sets/`
- `config.py` — all constants centralised (models, paths, chunk sizes, thresholds)
- `app.py` — minimal Streamlit UI shell with two states:
  - State 1: onboarding (no KB yet) — Build button, PDF count
  - State 2: full interface — tabs, query input, sidebar stats
- `requirements.txt` — pinned dependencies
- `.env.example` — API key template
- Colorblind-friendly palette: purple primary, blue grading, no red

---

## Step (b) — Ingestion

- `src/ingest.py` — PDF loading via LangChain's PyPDFLoader
- Four section buckets extracted per paper:
  - `abstract` — capped at 3000 chars
  - `introduction` — last 1500 chars searched for aim/hypothesis signal
    words; falls back to last 700 chars
  - `discussion` — full text up to stop heading
  - `limitations_future` — explicit section or fallback sentence extraction
    from discussion
- Regex patterns: ABSTRACT_PATTERN, INTRODUCTION_PATTERN, METHODS_PATTERN,
  DISCUSSION_PATTERN, CONCLUSIONS_PATTERN, LIMITATIONS_PATTERN, STOP_PATTERN
- Stop headings: References, Acknowledgments, Supplementary Material,
  Author Contributions, Funding, Declaration of Interest, Publication History,
  Editorial Note, Abbreviations, Glossary, About the Authors, and more
- Inline reference catcher: `STOP_PATTERN_INLINE_REFS` — catches
  "References 1." mid-paragraph
- Unnumbered reference list catcher: `STOP_PATTERN_REFLIST` — catches
  consecutive author-year citation entries without a heading
- PDF artifact cleaning: strips journal headers/footers and figure captions
- `PaperStatus` dataclass tracks per-paper ingestion outcome
- Ingestion statuses persisted to `chroma_db_statuses.json` including
  temporal tag counts — restored on app restart

---

## Step (c) — Retrieval

- `src/retrievers.py` — hybrid search:
  - Semantic search via ChromaDB cosine similarity (`SEMANTIC_TOP_K`)
  - Keyword search via BM25 (`BM25_TOP_K`)
  - Reciprocal Rank Fusion (RRF, K=60) combines both rankings
  - Returns top `FINAL_TOP_K` chunks
- `ScoredChunk` dataclass carries: text, paper_id, section_type,
  semantic_score, bm25_score, rrf_score
- `get_all_chunks()` — returns all chunks from ChromaDB for pipeline use
- `retrieve_limitations()` — targeted retrieval for limitations/discussion

---

## Step (d) — Tool 1: Metadata Extraction

- `src/tools.py` → `run_metadata_extraction()`
- One LLM call per paper extracts 8 fields: topic, hypothesis,
  research_question, methods, key_findings, discussion, limitations,
  future_recommendations
- Extraction prompt updated to extract limitations from discussion when
  no standalone section exists
- Build-time metadata extraction added: key_findings + hypothesis
  extracted at KB build time to ground temporal tagging

---

## Step (e) — Temporal Tagging

- `src/ingest.py` → `_tag_chunks_past_future()`
- Hybrid tagging approach:
  - `introduction` chunks → always "past" (deterministic)
  - `limitations_future` chunks → always "future" (deterministic)
  - `abstract` / `discussion` → cosine similarity to reference descriptions
  - Metadata grounding: if chunk more similar to paper's own key_findings
    + hypothesis than to either reference → tagged "past"
- `temporal_lean` metadata field: "past" | "future" | "neutral"
- `temporal_source` metadata field: "section_type" | "metadata" | "embedding"
- `TEMPORAL_REFERENCES` strengthened with meta-analysis specific language
- Tag counts displayed in Ingestion tab: past · future · neutral

---

## Step (f) — Tool 2: Gap Analysis + Hypothesis Generation

- `src/tools.py` → `run_gap_analysis()`
- Past summary (3 bullets) from: past-tagged chunks + key_findings +
  tested hypotheses from metadata
- Future summary (3 bullets) from: future-tagged chunks +
  future_recommendations + limitations from metadata
- Literature gap score: `1 - cosine(past_summary, future_summary)`
  with three-category grading in blue tones
- Single hypothesis generated from past + future summaries + gap score
- `src/generate.py` — LangChain prompts:
  - `PAST_SUMMARY_PROMPT` / `FUTURE_SUMMARY_PROMPT`
  - `SINGLE_HYPOTHESIS_PROMPT`
  - `PLAUSIBILITY_PROMPT` (LLM-as-judge, 6 dimensions)

---

## Step (g) — Originality Scoring

- `src/originality.py` → `score_originality_against_summary()`
- Originality = `1 - cosine(hypothesis, past_summary)`
- Three-category grading: Very original / Moderately original / Less original
- Same blue-tone palette used for all graded outputs (gap, originality, PubMed)
- `grade_similarity()` shared function for all three contexts

---

## Step (h) — Scientific Plausibility Judge

- `src/generate.py` → `judge_scientific_plausibility()`
- LLM scores hypothesis on 6 dimensions (1-5 each):
  1. Novelty
  2. Testability
  3. Mechanistic coherence
  4. Citation traceability
  5. Conflict awareness
  6. Usefulness for future study design
- Average score shown in hypothesis card with blue-tone coloring
- If score < 2.0: advisory note shown to user
- Individual dimension scores stored in `evaluation_results.json`

---

## Step (i) — Tool 4: PubMed Freshness Check

- `src/pubmed_tool.py` → `check_pubmed_freshness()`
- Direct NCBI E-utilities API (esearch + efetch) — no API key needed
- Search uses user topic (not hypothesis text) for broader keyword match
- Top 5 abstracts fetched, embedded, compared to hypothesis via cosine
- Same three-category grading in blue tones
- Shows: papers found count, papers compared, similarity score
- Papers listed when similarity ≥ 0.8
- Opt-in button in Generated Hypothesis tab

---

## Step (j) — Query Guard + Safety

- `src/query_guard.py` → `validate_query()`
- Checks: prompt injection, harmful intent, gibberish, off-topic
- System role defined internally (not shown to user)
- Fires before pipeline runs — warning shown under query input

---

## Step (k) — Cost Tracking

- `src/cost_tracking.py` — tiktoken-based token counting
- `SessionCostTracker` singleton in Streamlit session state
- Tracks: input tokens, output tokens, estimated USD cost per call
- Call log shown in sidebar (expandable)
- Pricing table in `config.py`

---

## Step (l) — PDF Export

- `src/exports.py` → `export_to_pdf()` using ReportLab
- Activated after hypothesis generated (sidebar button)
- Contains: query, gap analysis table, hypothesis, originality grade,
  plausibility score, PubMed results (if run), supporting papers,
  suggested approach

---

## Step (m) — Evaluation

- `evaluate.py` — standalone benchmark script (bypasses query guard)
- `data/eval_sets/benchmark.json` — 12 test cases:
  - 4 direct retrieval
  - 3 cross-paper synthesis
  - 2 hypothesis generation
  - 2 negative controls (out-of-corpus neuroscience)
  - 1 conflicting evidence
- RAGAS v0.2.x: faithfulness + answer_relevancy
- LLM-as-judge: 6-dimension plausibility scores per case
- Pass/fail evaluation per case
- Results saved to `evaluation_results.json`
- Results: 10/12 pass rate (83%), RAGAS faithfulness 0.145
  (expected low — hypothesis generator synthesises beyond retrieved text),
  answer_relevancy 0.771, plausibility avg 4.2/5

---
