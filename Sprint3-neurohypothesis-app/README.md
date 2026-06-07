# Neurohypothesis

A neuroscience hypothesis generation agent that produces evidence-grounded research hypotheses from your uploaded PDFs, live PubMed search, or both — with a human-in-the-loop rating step after every hypothesis.

Built with **Streamlit · LangGraph · OpenAI · ChromaDB** by the assistance of **Anthropic Claude**

---

## What it does

Given a research question (e.g. *"How does circadian rhythm disruption affect amyloid beta aggregation?"*) the agent generates **up to 6 hypotheses, one at a time**. After each one you rate it 1–5 and either **Continue** to the next or **Finish** to export — a Human-in-the-Loop checkpoint on every card.

Every hypothesis is scored on:
- **Originality** — cosine distance from a summary of your own library / retrieved past work.
- **Plausibility** — a 6-dimension LLM judge (novelty, testability, mechanistic coherence, citation traceability, conflict awareness, usefulness).
- **Past → future gap** — `1 − cosine(past, future)` for the evidence pair the hypothesis is built on.
- **Novelty vs PubMed** — the topic is translated to MeSH terms, PubMed is searched over the **last 25 years**, abstracts are embedded, and novelty is `1 − max cosine` to the nearest published paper (cached per topic).
- **Contradictory evidence** — a PubMed search → LLM-verification pass flags published work that opposes the hypothesis.

At the end it **exports a PDF report** with APA-style citations.

### Three execution paths

- **PubMed only** *(default)* — picks a primary category from six neuroscience method categories, retrieves papers per category with a 5-tier Boolean + MeSH strategy and cosine re-ranking, then generates hypotheses (H1 anchored to the primary category; H2–H6 each pair it with a different complementary category).
- **PDFs only** — works entirely within the papers you upload. It summarises each paper, builds **past → future gap pairs** from the summary bullets, and generates one hypothesis per pair, **incrementally up to 6**. Pairs are selected to stay relevant yet divergent (cosine band) and to maximise variety (tiered selection + MMR), with a diversity gate that regenerates a hypothesis if it is too similar to a previous one. This path is intentionally *not* organised by the six categories — it stays inside your literature.
- **Combined** — merges PubMed metadata with your PDF chunks as the evidence base for each hypothesis.

> **Note on the PDF-only ceiling:** with only a few uploaded papers there are a limited number of distinct gap pairs, so the first few hypotheses are the most distinct and later ones become recombinations. More papers → more distinct hypotheses.

---

## Quick start

### Requirements
- Python 3.12+
- An OpenAI API key

### Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/n-ayyildiz/AI-Eng-Turing.git
cd AI-Eng-Turing/Sprint3-neurohypothesis-app

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -e .

# 4. Configure secrets
cp .env.example .env
# Open .env and add your OpenAI API key:
#   OPENAI_API_KEY=sk-...
# NCBI_API_KEY is optional — raises PubMed rate limit from 3 to 10 req/s
```

### Run

```bash
streamlit run app.py
```

Open the browser tab Streamlit prints (usually http://localhost:8501).

---

## How to use

The input screen is a left-to-right, three-step flow:

1. **Research question** — enter a topic in the text box (the more specific, the better).
2. **How to answer it** — choose **PubMed only** (default), **PDFs only**, or **PDFs + PubMed**. When the choice involves PDFs, an uploader appears (up to 3 PDFs); PubMed-only needs no upload.
3. **Generate** — the **Generate Hypothesis** button stays disabled until you're ready (a question is entered, and at least one PDF is uploaded if the chosen path needs it).

Then, for each hypothesis: a live progress stream shows what the agent is doing, the card appears with its scores and gap analysis, and you **rate it 1–5** (optional comment) before clicking **Continue** for the next or **Finish/Stop** to end and download the **PDF report**.

> Uploaded PDFs must contain selectable text. Scanned/image-only PDFs cannot be read without OCR (not currently included), so their titles and metadata will be blank.

---

## Deploying to Streamlit Community Cloud

1. Push your code to a public GitHub repo (ensure `.env` and `.streamlit/secrets.toml` are gitignored — they are by default).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo.
3. Under **App settings → Secrets**, paste the contents of `secrets.toml.template` with your real values filled in.
4. Deploy.

Streamlit Cloud reads `requirements.txt` for dependencies.

---

## Supabase session logging *(optional)*

If `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set, completed sessions are logged to a `neurohypothesis_sessions` table. Run the SQL in `supabase_setup.sql` once to create the table, and `supabase_add_columns.sql` to add token-tracking columns.

Without these variables the app works fully — logging is silently skipped.

---

## Project structure

```
neurohypothesis/
├── app.py                          # Streamlit UI — all phases and hypothesis cards
├── config.py                       # All settings, constants, category definitions
├── pyproject.toml                  # Project metadata and dependencies (hatchling)
├── requirements.txt                # Flat dependency list for Streamlit Cloud
├── .env.example                    # Template — copy to .env and fill in keys
├── secrets.toml.template           # Template for .streamlit/secrets.toml
├── supabase_setup.sql              # One-time table creation for session logging
├── supabase_add_columns.sql        # Adds token/cost columns to existing table
├── src/
│   ├── agent_state.py              # AgentState TypedDict for LangGraph
│   ├── cost_tracking.py            # Per-node token and cost accounting
│   ├── db.py                       # SQLite persistence layer
│   ├── feedback.py                 # Supabase session logger
│   ├── utils.py                    # Embeddings, cosine similarity, helpers
│   ├── engine/
│   │   ├── chunking.py             # PDF chunking (title block preserved) + PubMed abstract ingestion
│   │   ├── exports.py              # PDF report generation (ReportLab)
│   │   ├── generate.py             # Summarisation, gap scoring, hypothesis generation (incl. Path A single-anchor)
│   │   ├── originality.py          # Cosine originality + gap-pair build / tiered-MMR selection / diversity gate
│   │   └── temporal_tagging.py     # Past / future chunk classification
│   ├── graph/
│   │   ├── graph.py                # LangGraph wiring and topology export
│   │   └── nodes.py                # All node functions (incl. n_a_generate, MeSH novelty check, N4c metadata)
│   └── tools/
│       ├── moderation.py           # Input validation and OpenAI moderation
│       ├── pubmed.py               # 5-tier Boolean search, MeSH translation, contradictory evidence
│       └── retriever.py            # Chroma vectorstore hybrid retrieval (BM25 + semantic)
└── tests/                          # Unit tests for deterministic components
```

---

## Architecture

```
Streamlit UI (question → path + upload → generate)
    └── LangGraph StateGraph
            ├── N1–N3      validate → parse topic → route path
            │
            ├── PATH A     N4a–c: local PDF ingest → chunk → embed
            │              N4c:   per-paper metadata extraction (real title/authors/journal/year + topic/findings/limitations/future)
            │              N9–N12: retrieve evidence → summarise past + future → gap
            │              n_a_generate (INCREMENTAL loop, 1 → 6):
            │                   build past→future gap pairs from summary bullets
            │                   (cosine band 0.35–0.70) → tiered selection + MMR
            │                   → one hypothesis per anchor pair
            │                   → diversity gate (regenerate once if ≥0.75 cosine to a prior)
            │
            ├── PATH B     N5_pick_primary → N5_order_categories
            │              → N5_per_category_retrieve (5-tier loop × 6 categories)
            │              → N5_embed_category_papers → validation gate
            │              N8–N16 loop: select primary + complementary pair → retrieve → summarise → gap → generate → score
            │
            ├── PATH C     as Path B, but N9 merges PubMed metadata + PDF chunks
            │
            └── Shared     N14–N15: originality + plausibility (6-dim LLM judge)
                           N16:     quality gate (regenerate up to 3×) + novelty vs PubMed (MeSH, 25 yr) + contradictory evidence
                           N17–N18: HITL — present card + collect rating (interrupt)
                           N19–N20: export PDF + persist session to SQLite
```

---

## Determinism

| Component | Setting |
|---|---|
| LLM seed | `LLM_SEED = 42` |
| Decision-making / judge temperature | `T = 0.0` |
| Hypothesis generation (PubMed / Combined) | `T = 0.0`, `seed = LLM_SEED + hyp_index` |
| Hypothesis generation (PDFs only) | `T = PATH_A_GEN_TEMPERATURE` (0.3), `seed = LLM_SEED + slot` (`+100` on diversity-gate retry) |
| Embeddings | `text-embedding-3-small` (deterministic by design) |

Each hypothesis slot is independently seeded, so a given position is reproducible while still allowing variety across slots. Runs with the same query can still differ because PubMed retrieval changes as the index updates.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Disclaimer

AI-generated hypotheses are research suggestions, not scientific facts. All outputs require expert validation before use in actual research.
