# cv_parser.py
# Handles CV file uploads — reads PDF and Word (.docx) files,
# extracts raw text, and uses OpenAI to summarise key information.

import os
import fitz  # PyMuPDF
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ----------------------------------------------------------------------
# FILE READERS
# ----------------------------------------------------------------------

def extract_text_from_pdf(file) -> str:
    """Extract plain text from an uploaded PDF file object."""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def extract_text_from_docx(file) -> str:
    """Extract plain text from an uploaded .docx file object."""
    document = Document(file)
    return "\n".join([para.text for para in document.paragraphs]).strip()


def extract_cv_text(uploaded_file) -> tuple[bool, str]:
    """
    Route to the correct extractor based on file type.
    Returns (success: bool, text_or_error: str).
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        try:
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                return False, "⚠️ Could not extract text from this PDF. It may be image-based or scanned."
            return True, text
        except Exception as e:
            return False, f"⚠️ Error reading PDF: {str(e)}"

    elif filename.endswith(".docx"):
        try:
            text = extract_text_from_docx(uploaded_file)
            if not text:
                return False, "⚠️ Could not extract text from this Word file. It may be empty."
            return True, text
        except Exception as e:
            return False, f"⚠️ Error reading Word file: {str(e)}"

    else:
        return False, "⚠️ Unsupported file type. Please upload a PDF or .docx file."


# ----------------------------------------------------------------------
# AI-POWERED CV SUMMARY
# ----------------------------------------------------------------------

CV_EXTRACTION_PROMPT = """
You are a CV analysis assistant. Extract and summarise the following information from the CV text below.
Return your response in this exact format:

**Name:** (candidate's name if found, otherwise "Not specified")
**Current/Most Recent Role:** (job title and company)
**Years of Experience:** (estimated total years)
**Technical Skills:** (comma-separated list)
**Soft Skills:** (comma-separated list, infer from experience descriptions if not explicit)
**Education:** (highest degree, field, institution)
**Key Achievements:** (2-3 most impressive bullet points)
**Notable Gaps or Areas to Develop:** (skills or experience that seem limited based on the CV)

Be concise. Do not invent information not present in the CV.
"""


def summarise_cv(cv_text: str, model: str = "gpt-4.1-mini") -> tuple[bool, str]:
    """
    Use OpenAI to extract structured information from raw CV text.
    Returns (success: bool, summary_or_error: str).
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CV_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Here is the CV:\n\n{cv_text[:6000]}"},
            ],
            temperature=0.2,  # Low temperature for consistent structured extraction
            max_tokens=600,
        )
        return True, response.choices[0].message.content
    except Exception as e:
        return False, f"⚠️ Error analysing CV: {str(e)}"


# ----------------------------------------------------------------------
# JOB DESCRIPTION ANALYSER
# ----------------------------------------------------------------------

JD_EXTRACTION_PROMPT = """
You are a job description analyst. Extract the key information from the job posting below.
Return your response in this exact format:

**Job Title:** (exact title from the posting)
**Seniority Level:** (e.g. Junior, Mid-level, Senior, Lead)
**Required Technical Skills:** (comma-separated list)
**Required Soft Skills:** (comma-separated list)
**Key Responsibilities:** (3-5 bullet points)
**Nice-to-Have Skills:** (comma-separated list, or "None specified")
**Interview Focus Areas:** (2-3 areas the interview is likely to test based on this role)

Be concise and accurate. Only include information present in the job description.
"""


def analyse_job_description(jd_text: str, model: str = "gpt-4.1-mini") -> tuple[bool, str]:
    """
    Use OpenAI to extract structured information from a job description.
    Returns (success: bool, summary_or_error: str).
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JD_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Here is the job description:\n\n{jd_text[:4000]}"},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return True, response.choices[0].message.content
    except Exception as e:
        return False, f"⚠️ Error analysing job description: {str(e)}"


# ----------------------------------------------------------------------
# GAP ANALYSIS
# ----------------------------------------------------------------------

GAP_ANALYSIS_PROMPT = """
You are a career coach. Compare the candidate's CV with the job description and identify:

1. **Matching Strengths** — skills/experience the candidate has that the job requires
2. **Skill Gaps** — skills the job requires that the candidate lacks or has limited experience in
3. **Interview Priority Areas** — the 3 most important topics to focus on in interview preparation, based on the gaps
4. **Confidence Level** — rate the candidate's fit for this role: Strong / Moderate / Needs Work

Be honest, specific, and constructive. This analysis will guide the interview preparation session.
"""


def analyse_gap(cv_summary: str, jd_summary: str, model: str = "gpt-4.1-mini") -> tuple[bool, str]:
    """
    Compare CV and job description to identify gaps and priorities.
    Returns (success: bool, analysis_or_error: str).
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GAP_ANALYSIS_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"CV Summary:\n{cv_summary}\n\n"
                        f"Job Description Summary:\n{jd_summary}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return True, response.choices[0].message.content
    except Exception as e:
        return False, f"⚠️ Error generating gap analysis: {str(e)}"
