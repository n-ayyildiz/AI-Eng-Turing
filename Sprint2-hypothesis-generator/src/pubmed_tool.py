"""
PubMed freshness check (Tool 4).

After the generated hypothesis is scored against the local library's
past_summary, this tool queries PubMed for recent publications on the
topic and compares them against the hypothesis via cosine similarity.

Purpose: catch hypotheses that are novel within the local library but
may already have been tested in recent literature the user does not
have on disk.

Architecture: deterministic function call (not agent-style).
Uses NCBI E-utilities API directly (esearch + efetch) — free, no API
key needed for the volumes this tool generates.

Public API:
    - check_pubmed_freshness(hypothesis_text, embeddings, topic) -> dict
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from langchain_openai import OpenAIEmbeddings

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class PubMedMatch:
    """One PubMed abstract retrieved + similarity to the hypothesis."""
    title: str
    year: str
    abstract: str
    url: str
    similarity: float


# =============================================================================
# Helpers
# =============================================================================

def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _grade_pubmed(similarity: float) -> dict:
    """
    Convert similarity to the three-category grade + label + purple color.

    Thresholds match the rest of the app (VERY_ORIGINAL_THRESHOLD=0.3,
    LESS_ORIGINAL_THRESHOLD=0.8).
    """
    labels = {
        "very":     "Very original",
        "moderate": "Moderately original",
        "less":     "Less original",
    }

    if similarity <= config.VERY_ORIGINAL_THRESHOLD:
        grade = "very"
    elif similarity >= config.LESS_ORIGINAL_THRESHOLD:
        grade = "less"
    else:
        grade = "moderate"

    return {
        "grade": grade,
        "label": labels[grade],
        "color": config.PUBMED_COLORS[grade],
    }


def _parse_pubmed_output(raw: str) -> list[dict]:
    """
    Parse LangChain's PubmedQueryRun output into a list of paper dicts.

    The tool returns a single concatenated string with fields like:
        Published: 2024-05-12
        Title: ...
        Copyright Information: ...
        Summary::
        ...abstract...

    Papers are separated by blank lines. This parser extracts what is
    needed (title, year, abstract) without relying on a specific format.
    """
    papers = []
    # Papers in the output are separated by double newlines
    blocks = re.split(r"\n\s*\n", raw.strip())

    for block in blocks:
        if not block.strip():
            continue

        title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", block)
        date_match = re.search(r"Published:\s*(\d{4})", block)
        summary_match = re.search(
            r"Summary::\s*(.+?)(?:\n\n|\Z)",
            block,
            re.DOTALL,
        )

        title = title_match.group(1).strip() if title_match else ""
        year = date_match.group(1) if date_match else ""
        abstract = summary_match.group(1).strip() if summary_match else ""

        # Construct a search-based PubMed URL using the title
        # (PMID isn't consistently extractable, and the user said
        # PMIDs aren't needed in the output anyway)
        if title:
            safe_title = title.replace(" ", "+")[:120]
            url = f"https://pubmed.ncbi.nlm.nih.gov/?term={safe_title}"
            papers.append({
                "title": title,
                "year": year,
                "abstract": abstract,
                "url": url,
            })

    return papers


# =============================================================================
# Main function
# =============================================================================

def check_pubmed_freshness(
    hypothesis_text: str,
    embeddings: OpenAIEmbeddings,
    topic: str = "",
) -> dict:
    """
    Query PubMed with the hypothesis, compute similarity against recent
    abstracts, return a graded freshness result.

    Returns a dict with keys:
        'status'       : "ok" | "no_results" | "error"
        'message'      : user-facing status message
        'grade'        : "very" | "moderate" | "less" (only when status=ok)
        'label'        : human-readable label
        'color'        : purple hex for UI
        'max_similarity' : float (only when status=ok)
        'matches'      : list[dict] — papers above PUBMED_SHOW_MATCHES_AT threshold
        'all_results'  : list[dict] — all compared papers (for debugging)

    Args:
        hypothesis_text: the generated hypothesis statement to check.
        embeddings: the OpenAI embeddings client.
        topic: optional user topic to combine with the hypothesis in the
               PubMed query. When provided, the topic terms are appended
               to the hypothesis in the search string — PubMed's relevance
               ranking uses both as signals, returning papers that best
               match the combined intent. Uses space separation (not AND)
               to avoid over-narrowing results.
    """
    if not hypothesis_text or not hypothesis_text.strip():
        return {
            "status": "error",
            "message": "No hypothesis text to check.",
        }

    # Use the topic for PubMed's keyword search (Step 1). The hypothesis
    # is compared to the returned abstracts via cosine similarity (Step 2).
    query_text = topic.strip() if topic and topic.strip() else hypothesis_text.strip()

    current_year = datetime.now().year
    start_year = current_year - config.PUBMED_YEARS_BACK

    # --- Direct NCBI E-utilities API (bypasses LangChain wrappers which
    #     proved unreliable — returning truncated/unparseable results). ---
    import requests
    import xml.etree.ElementTree as ET

    # Step A: Search PubMed for PMIDs matching the query
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": f"{query_text} AND {start_year}:{current_year}[dp]",
            "retmax": config.PUBMED_TOP_N,
            "sort": "relevance",
            "retmode": "json",
        }
        search_resp = requests.get(search_url, params=search_params, timeout=15)
        search_resp.raise_for_status()
        search_data = search_resp.json()
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        total_count = int(search_data.get("esearchresult", {}).get("count", 0))
    except Exception as e:
        logger.warning(f"PubMed search failed: {e}")
        return {
            "status": "error",
            "message": f"PubMed is not responding right now.",
            "query_used": query_text,
        }

    if not pmids:
        return {
            "status": "no_results",
            "message": "No recent matches found in PubMed.",
            "query_used": query_text,
            "total_found": 0,
        }

    # Step B: Fetch article details (title, abstract, year) for those PMIDs
    try:
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
        }
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=15)
        fetch_resp.raise_for_status()
        root = ET.fromstring(fetch_resp.text)
    except Exception as e:
        logger.warning(f"PubMed fetch failed: {e}")
        return {
            "status": "error",
            "message": f"PubMed is not responding right now.",
            "query_used": query_text,
        }

    # Step C: Parse XML into structured paper dicts
    papers = []
    for article in root.findall(".//PubmedArticle"):
        # Title
        title_el = article.find(".//ArticleTitle")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

        # Year
        year_el = article.find(".//PubDate/Year")
        year = year_el.text if year_el is not None else ""
        if not year:
            medline_el = article.find(".//MedlineDate")
            if medline_el is not None and medline_el.text:
                ym = re.search(r"(\d{4})", medline_el.text)
                year = ym.group(1) if ym else ""

        # PMID
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        # Abstract
        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(
            (part.text or "") for part in abstract_parts
        ).strip()

        if title:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            papers.append({
                "title": title,
                "year": year,
                "abstract": abstract,
                "url": url,
            })

    if not papers:
        return {
            "status": "no_results",
            "message": "No recent matches found in PubMed.",
            "query_used": query_text,
            "total_found": total_count,
        }

    # Embed the hypothesis and all abstracts
    try:
        texts = [hypothesis_text] + [p["abstract"] or p["title"] for p in papers]
        vecs = embeddings.embed_documents(texts)
        hyp_vec = vecs[0]
        paper_vecs = vecs[1:]
    except Exception as e:
        logger.error(f"Embedding failed during PubMed check: {e}")
        return {
            "status": "error",
            "message": "PubMed is not responding right now.",
        }

    # Score each paper
    scored = []
    for paper, vec in zip(papers, paper_vecs):
        sim = _cosine(hyp_vec, vec)
        scored.append({
            "title": paper["title"],
            "year": paper["year"],
            "url": paper["url"],
            "similarity": sim,
        })

    scored.sort(key=lambda p: p["similarity"], reverse=True)
    max_sim = scored[0]["similarity"]
    grade_info = _grade_pubmed(max_sim)

    # Matches above the show-threshold (only shown in UI when similarity is high)
    matches = [p for p in scored if p["similarity"] >= config.PUBMED_SHOW_MATCHES_AT]

    logger.info(
        f"PubMed freshness check: {len(papers)} papers checked, "
        f"max_sim={max_sim:.3f}, grade={grade_info['label']}, "
        f"matches_above_threshold={len(matches)}"
    )

    return {
        "status": "ok",
        "message": f"Checked {len(papers)} recent PubMed abstracts.",
        "query_used": query_text,
        "total_found": total_count,
        "papers_compared": len(papers),
        "grade": grade_info["grade"],
        "label": grade_info["label"],
        "color": grade_info["color"],
        "max_similarity": max_sim,
        "matches": matches,
        "all_results": scored,
    }
