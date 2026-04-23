# Hypothesis Generator

A Streamlit application that reads a local corpus of neuroscience papers,
builds a searchable knowledge base with hybrid RAG, analyses gaps between
past findings and future recommendations, and generates novel testable
research hypotheses grounded in the literature.

**Status:** Complete. All core and some optional features implemented.

---

## Quick Start

```bash
# 1. Activate the virtual environment (from project root)
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file from the template
cp .env.example .env
# Open .env and add your OPENAI_API_KEY

# 4. Drop your PDFs into data/papers/

# 5. Run the app
python -m streamlit run app.py
```

The app opens at http://localhost:8501.

---

## First-time environment setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Every new terminal needs `source venv/bin/activate` before running the app.

---

## Optional: log file output

Add to `.env` to write all INFO logs to a file:

```
LOG_FILE=app.log
ANONYMIZED_TELEMETRY=False
```

---

## Running the evaluation

```bash
python evaluate.py
```

Runs a 12-case benchmark covering direct retrieval, cross-paper synthesis,
hypothesis generation, and negative controls. Results saved to
`evaluation_results.json`.

---

## Project Structure

```
project/
├── app.py                     # Streamlit entry point
├── config.py                  # All constants: models, paths, thresholds
├── evaluate.py                # Standalone benchmark evaluation script
├── requirements.txt           # Pinned dependencies
├── .env                       # Secrets — not committed
├── .env.example               # Template for .env
├── .gitignore
├── README.md
│
├── src/
│   ├── ingest.py              # PDF loading, section detection, chunking,
│   │                          # temporal tagging (past/future/neutral)
│   ├── retrievers.py          # Semantic + BM25 + RRF hybrid retrieval
│   ├── tools.py               # Tool 1: metadata extraction
│   │                          # Tool 2: gap analysis + hypothesis generation
│   ├── pubmed_tool.py         # Tool 4: PubMed freshness check (NCBI E-utilities)
│   ├── generate.py            # LangChain prompts and chains
│   ├── originality.py         # Cosine similarity grading for originality
│   ├── query_guard.py         # Input validation and safety checks
│   ├── exports.py             # PDF export via ReportLab
│   ├── cost_tracking.py       # tiktoken-based token and cost logging
│   └── __init__.py
│
├── data/
│   ├── papers/                # Local PDF corpus goes here
│   └── eval_sets/
│       └── benchmark.json     # 12-case evaluation benchmark
│
└── chroma_db/                 # Auto-created by ChromaDB at runtime
```

---

## Tech Stack

| Component | Choice |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain (light — manual chains) |
| Main LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB (local PersistentClient) |
| Sparse retrieval | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF, manual Python) |
| External API | PubMed via NCBI E-utilities (free, no key needed) |
| Evaluation | RAGAS v0.2.x + LLM-as-judge (6 dimensions) |
| PDF export | ReportLab |

---

## Pipeline Overview

1. **Ingestion** — PDFs are loaded, cleaned, and split into four section
   buckets: abstract, introduction, discussion, limitations_future. Each
   chunk is embedded and tagged as past/future/neutral using a hybrid
   approach: deterministic by section type for introduction and
   limitations_future, cosine similarity for abstract and discussion,
   with metadata grounding for additional accuracy.

2. **Retrieval** — Hybrid search combines semantic (ChromaDB) and keyword
   (BM25) results via RRF fusion. Returns the top-K most relevant chunks.

3. **Metadata extraction (Tool 1)** — One LLM call per paper extracts
   structured fields: topic, hypothesis, research question, methods,
   key findings, limitations, future recommendations.

4. **Gap analysis (Tool 2)** — Past-tagged chunks + key findings +
   tested hypotheses are summarised into 3 past bullets. Future-tagged
   chunks + limitations + future recommendations are summarised into
   3 future bullets. Literature gap score computed as
   `1 - cosine(past_summary, future_summary)`.

5. **Hypothesis generation** — One LLM call generates a single novel
   testable hypothesis from the past/future summaries and gap score.

6. **Originality scoring** — Cosine similarity between the hypothesis
   and past summary. Three-category grading: Very original / Moderately
   original / Less original.

7. **Scientific plausibility judge** — LLM scores the hypothesis on
   6 dimensions (novelty, testability, mechanistic coherence, citation
   traceability, conflict awareness, usefulness). Average shown to user.

8. **PubMed check (Tool 4, opt-in)** — Searches NCBI PubMed for recent
   papers (last 5 years) on the topic. Compares abstracts to hypothesis
   via cosine similarity. Same three-category grading in blue tones.

---

## Evaluation Framework

Four-layer evaluation:

- **Layer A** — Extraction quality: manually verified for all 5 papers
- **Layer B** — Retrieval quality: chunk-level RRF scores logged per benchmark case
- **Layer C** — Grounded generation: RAGAS metrics (faithfulness, answer relevancy)
- **Layer D** — Hypothesis quality: LLM-as-judge on 6 scientific dimensions

Run: `python evaluate.py`

---

## ⚠️ Disclaimer

All hypotheses are AI-generated research suggestions, not scientific facts.
Every output requires expert validation before use in actual research.
