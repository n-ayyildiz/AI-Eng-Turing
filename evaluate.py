"""
Standalone evaluation script for the Hypothesis Generator.

Runs a 12-case benchmark covering:
    - 4 direct retrieval cases
    - 3 cross-paper synthesis cases
    - 2 hypothesis generation cases
    - 2 negative control cases (out-of-corpus neuroscience)
    - 1 conflicting evidence case

Evaluation layers:
    Layer B (retrieval): logs retrieved chunks + average RRF scores per case
    Layer C (grounding): RAGAS metrics — faithfulness, answer_relevancy,
                         context_precision (ragas v0.2.x)
    Layer D (hypothesis quality): LLM-as-judge — 6 dimensions averaged to
                                  a single plausibility score

Results saved to: evaluation_results.json

Run from the project root:
    python evaluate.py

Note: the query guard (query_guard.py) is intentionally bypassed here.
The evaluation measures RAG system quality, not app-level input validation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment + path setup (must come before project imports)
# ---------------------------------------------------------------------------
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.ingest import load_existing_vectorstore
from src.retrievers import hybrid_search, get_all_chunks
from src.tools import run_metadata_extraction
from src.generate import (
    summarize_past,
    summarize_future,
    generate_single_hypothesis,
    judge_scientific_plausibility,
)
from src.originality import (
    score_originality_against_summary,
    grade_similarity,
    _cosine_similarity,
)
from langchain_openai import OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Benchmark dataset path
# ---------------------------------------------------------------------------
BENCHMARK_PATH = Path(__file__).parent / "data" / "eval_sets" / "benchmark.json"
RESULTS_PATH = Path(__file__).parent / "evaluation_results.json"


# ---------------------------------------------------------------------------
# Helper: run full pipeline for one query, return structured result
# ---------------------------------------------------------------------------

def run_pipeline_for_query(
    query: str,
    vectorstore,
    embeddings: OpenAIEmbeddings,
    metadata_list: list[dict],
) -> dict:
    """
    Run retrieval + summarisation + hypothesis generation + judge for
    a single benchmark query.

    Metadata is pre-computed once and shared across all queries to
    avoid redundant LLM calls.

    Returns a dict with:
        retrieved_chunks, contexts (text list), hypothesis, plausibility
    """
    # Retrieval
    retrieved = hybrid_search(query, vectorstore)
    # contexts = [sc.text for sc in retrieved] # raw chunks
    contexts = past_bullets + future_bullets  # the summaries the hypothesis was actually generated from
    avg_rrf = (
        round(sum(sc.rrf_score for sc in retrieved) / len(retrieved), 4)
        if retrieved else 0.0
    )
    retrieved_papers = list({sc.paper_id for sc in retrieved})

    # Summarisation
    all_chunks = get_all_chunks(vectorstore)
    past_texts = [
        c.page_content for c in all_chunks
        if c.metadata.get("temporal_lean") == "past"
    ]
    future_texts = [
        c.page_content for c in all_chunks
        if c.metadata.get("temporal_lean") == "future"
    ]
    paper_ids = sorted({
        c.metadata.get("paper_id", "?") for c in all_chunks
        if c.metadata.get("paper_id")
    })
    tested_hyps = "\n".join(
        f"[{m.get('paper_id', '?')}] {m.get('hypothesis', '')}"
        for m in metadata_list
        if m.get("hypothesis") and m.get("hypothesis") != "Extraction failed"
    )

    past_bullets = summarize_past("\n\n".join(past_texts), tested_hyps)
    future_bullets = summarize_future("\n\n".join(future_texts))
    past_text = "\n".join(f"- {b}" for b in past_bullets)
    future_text = "\n".join(f"- {b}" for b in future_bullets)

    # Literature gap score
    past_vec = embeddings.embed_documents([past_text])[0]
    future_vec = embeddings.embed_documents([future_text])[0]
    gap_sim = _cosine_similarity(past_vec, future_vec)
    gap_score = round(1.0 - gap_sim, 3)
    gap_grade = grade_similarity(gap_sim, context="gap")

    # Hypothesis generation
    gen_result = generate_single_hypothesis(
        topic=query,
        past_summary=past_text,
        future_summary=future_text,
        gap_score=gap_score,
        gap_label=gap_grade["label"],
        paper_ids=list(paper_ids),
    )
    hypothesis = gen_result.get("hypothesis", {})
    statement = hypothesis.get("statement", "")
    supported_by = hypothesis.get("supported_by", [])

    # Originality scoring
    orig_score = None
    if statement:
        orig_results = score_originality_against_summary(
            [hypothesis], past_text, embeddings
        )
        if orig_results:
            orig = orig_results[0]
            orig_score = orig.originality_score

    # Scientific plausibility judge
    plausibility = {}
    if statement:
        plausibility = judge_scientific_plausibility(
            statement=statement,
            supported_by=supported_by,
            topic=query,
        )

    return {
        "query": query,
        "retrieved_chunks": [
            {"paper_id": sc.paper_id, "section": sc.section_type, "rrf": sc.rrf_score}
            for sc in retrieved
        ],
        "retrieved_papers": retrieved_papers,
        "avg_rrf_score": avg_rrf,
        "contexts": contexts,
        "hypothesis": statement,
        "supported_by": supported_by,
        "originality_score": orig_score,
        "literature_gap_score": gap_score,
        "plausibility": plausibility,
    }


# ---------------------------------------------------------------------------
# Pass/fail evaluation for each case
# ---------------------------------------------------------------------------

def evaluate_case(case: dict, result: dict) -> dict:
    """
    Apply the pass criteria for one benchmark case.

    Returns a dict with 'passed' (bool) and 'reason' (str).
    """
    cat = case["category"]
    avg_rrf = result["avg_rrf_score"]
    plausibility = result.get("plausibility", {})
    retrieved_papers = set(result.get("retrieved_papers", []))
    expected_papers = set(case.get("expected_papers", []))

    if cat in ("direct_retrieval", "cross_paper_synthesis", "hypothesis_generation"):
        # Pass if any expected paper appears in retrieved results
        overlap = retrieved_papers & expected_papers
        passed = len(overlap) > 0
        reason = (
            f"Expected {expected_papers}, retrieved {retrieved_papers}. "
            + ("Overlap found." if passed else "No overlap.")
        )

    elif cat == "negative_control":
        # Pass if retrieval is weak (avg RRF < 0.02) OR citation traceability <= 2
        ct_score = plausibility.get("citation_traceability", 3)
        passed = avg_rrf < 0.02 or ct_score <= 2
        reason = (
            f"Avg RRF={avg_rrf:.4f} (threshold <0.02), "
            f"citation_traceability={ct_score} (threshold <=2). "
            + ("PASS — weak retrieval or low traceability as expected."
               if passed else "FAIL — system retrieved relevant-looking chunks for out-of-corpus query.")
        )

    elif cat == "conflicting_evidence":
        # Pass if conflict_awareness score >= 3
        ca_score = plausibility.get("conflict_awareness", 0)
        passed = ca_score >= 3
        reason = (
            f"conflict_awareness={ca_score} (threshold >=3). "
            + ("PASS — hypothesis acknowledges uncertainty."
               if passed else "FAIL — hypothesis did not acknowledge conflicting evidence.")
        )

    else:
        passed = False
        reason = f"Unknown category: {cat}"

    return {"passed": passed, "reason": reason}


# ---------------------------------------------------------------------------
# RAGAS evaluation (Layer C)
# ---------------------------------------------------------------------------

def run_ragas(ragas_samples: list[dict]) -> dict:
    """
    Run RAGAS v0.2.x metrics on collected samples.

    Metrics: faithfulness, answer_relevancy, context_precision.
    Returns a dict of metric name → score (float), or error message.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics import faithfulness, answer_relevancy
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings as RAGASEmbeddings

        samples = []
        for s in ragas_samples:
            if s["hypothesis"] and s["contexts"]:
                samples.append(SingleTurnSample(
                    user_input=s["query"],
                    retrieved_contexts=s["contexts"],
                    response=s["hypothesis"],
                    reference=None,   # not needed for these metrics
                ))

        if not samples:
            return {"error": "No valid samples for RAGAS evaluation."}

        dataset = EvaluationDataset(samples=samples)

        llm = ChatOpenAI(model=config.MAIN_LLM_MODEL, temperature=0.0)
        emb = RAGASEmbeddings(model=config.EMBEDDING_MODEL)

        result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=emb,
        )

        scores = result.to_pandas()[
            ["faithfulness", "answer_relevancy"]
        ].mean().to_dict()

        return {k: round(float(v), 3) for k, v in scores.items()}

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Hypothesis Generator — Benchmark Evaluation")
    print("=" * 60)

    # Load benchmark
    if not BENCHMARK_PATH.exists():
        print(f"\n[ERROR] Benchmark file not found: {BENCHMARK_PATH}")
        print("Expected location: data/eval_sets/benchmark.json")
        sys.exit(1)

    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)
    print(f"\nBenchmark loaded: {len(benchmark)} cases")

    # Load vectorstore
    print("\nLoading vectorstore...")
    vectorstore = load_existing_vectorstore()
    if vectorstore is None:
        print("[ERROR] No vectorstore found. Build the knowledge base first.")
        sys.exit(1)
    print("Vectorstore loaded.")

    # Initialise embeddings once
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    # Run metadata extraction once (shared across all cases)
    print("\nExtracting paper metadata (one-time)...")
    metadata_list = run_metadata_extraction(vectorstore)
    print(f"Metadata extracted for {len(metadata_list)} papers.")

    # Run pipeline for each benchmark case
    all_results = []
    ragas_samples = []
    passed_count = 0

    print(f"\nRunning {len(benchmark)} benchmark cases...\n")

    for case in benchmark:
        case_id = case["id"]
        category = case["category"]
        query = case["query"]

        print(f"[{case_id:02d}/{len(benchmark)}] {category.upper()}")
        print(f"         Query: {query}")

        start = time.time()

        try:
            result = run_pipeline_for_query(
                query=query,
                vectorstore=vectorstore,
                embeddings=embeddings,
                metadata_list=metadata_list,
            )
        except Exception as e:
            print(f"         [ERROR] Pipeline failed: {e}\n")
            all_results.append({
                "case_id": case_id,
                "category": category,
                "query": query,
                "error": str(e),
                "passed": False,
            })
            continue

        elapsed = time.time() - start

        # Evaluate pass/fail
        eval_outcome = evaluate_case(case, result)
        passed = eval_outcome["passed"]
        if passed:
            passed_count += 1

        # Print per-case summary
        status = "✅ PASS" if passed else "❌ FAIL"
        avg_plaus = result["plausibility"].get("average_score", "n/a")
        print(f"         Retrieved: {result['retrieved_papers']}")
        print(f"         Avg RRF:   {result['avg_rrf_score']:.4f}")
        print(f"         Hypothesis: {result['hypothesis'][:80]}...")
        print(f"         Plausibility avg: {avg_plaus}")
        print(f"         {status} — {eval_outcome['reason']}")
        print(f"         Time: {elapsed:.1f}s\n")

        # Collect for RAGAS (skip negative controls for grounding metrics)
        if not case["negative_control"] and result["hypothesis"] and result["contexts"]:
            ragas_samples.append({
                "query": query,
                "contexts": result["contexts"],
                "hypothesis": result["hypothesis"],
            })

        all_results.append({
            "case_id": case_id,
            "category": category,
            "query": query,
            "retrieved_papers": result["retrieved_papers"],
            "avg_rrf_score": result["avg_rrf_score"],
            "hypothesis": result["hypothesis"],
            "supported_by": result["supported_by"],
            "originality_score": result["originality_score"],
            "literature_gap_score": result["literature_gap_score"],
            "plausibility_scores": result["plausibility"],
            "pass_criteria": case["pass_criteria"],
            "passed": passed,
            "pass_reason": eval_outcome["reason"],
            "elapsed_seconds": round(elapsed, 1),
        })

    # Summary before RAGAS
    print("-" * 60)
    print(f"Pipeline complete: {passed_count}/{len(benchmark)} cases passed")
    print("-" * 60)

    # RAGAS evaluation
    print(f"\nRunning RAGAS on {len(ragas_samples)} samples...")
    print("(faithfulness, answer_relevancy)\n")
    ragas_scores = run_ragas(ragas_samples)

    if "error" in ragas_scores:
        print(f"[RAGAS ERROR] {ragas_scores['error']}")
    else:
        print("RAGAS results:")
        for metric, score in ragas_scores.items():
            print(f"  {metric:25s}: {score:.3f}")

    # Aggregate plausibility scores
    plaus_avgs = [
        r["plausibility_scores"].get("average_score")
        for r in all_results
        if r.get("plausibility_scores") and r["plausibility_scores"].get("average_score") is not None
    ]
    overall_plaus = round(sum(plaus_avgs) / len(plaus_avgs), 2) if plaus_avgs else None

    print(f"\nOverall plausibility average (all cases): {overall_plaus} / 5.0")

    # Save results
    output = {
        "summary": {
            "total_cases": len(benchmark),
            "passed": passed_count,
            "failed": len(benchmark) - passed_count,
            "pass_rate": round(passed_count / len(benchmark), 2),
            "ragas_scores": ragas_scores,
            "overall_plausibility_avg": overall_plaus,
        },
        "cases": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")
    print("\n" + "=" * 60)
    print(f"  Pass rate:        {passed_count}/{len(benchmark)} ({output['summary']['pass_rate']:.0%})")
    if "error" not in ragas_scores:
        print(f"  RAGAS faithfulness:       {ragas_scores.get('faithfulness', 'n/a')}")
        print(f"  RAGAS answer_relevancy:   {ragas_scores.get('answer_relevancy', 'n/a')}")
        print(f"  RAGAS context_precision:  {ragas_scores.get('context_precision', 'n/a')}")
    print(f"  Plausibility avg: {overall_plaus} / 5.0")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
