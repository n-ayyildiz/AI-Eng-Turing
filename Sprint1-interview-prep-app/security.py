# security.py
# Security guards to protect the app from misuse and prompt injection.
# Each function returns (is_safe: bool, reason: str).

import re

MAX_INPUT_LENGTH = 1500
MIN_INPUT_LENGTH = 2
MAX_FILE_SIZE_MB = 5
ALLOWED_EXTENSIONS = [".pdf", ".docx"]

INJECTION_PATTERNS = [
    r"ignore (all |previous |above |prior )?instructions",
    r"disregard (all |previous |above |prior )?instructions",
    r"forget (all |previous |above |prior )?instructions",
    r"you are now",
    r"act as (a |an )?(?!interview)",
    r"new persona",
    r"override",
    r"system prompt",
    r"reveal (your |the )?(prompt|instructions|system)",
    r"print (your |the )?(prompt|instructions|system)",
    r"what (are|were) your instructions",
    r"jailbreak",
    r"pretend (you are|to be)",
    r"roleplay as",
    r"DAN",
    r"developer mode",
    r"sudo",
]


def check_length(text: str) -> tuple[bool, str]:
    length = len(text.strip())
    if length < MIN_INPUT_LENGTH:
        return False, "⚠️ Your message is too short. Please type a proper answer."
    if length > MAX_INPUT_LENGTH:
        return False, (
            f"⚠️ Your message is too long ({length} characters). "
            f"Please keep responses under {MAX_INPUT_LENGTH} characters."
        )
    return True, ""


def check_injection(text: str) -> tuple[bool, str]:
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return False, (
                "🚫 Your message appears to contain a prompt injection attempt. "
                "This app is designed strictly for interview preparation."
            )
    return True, ""


def check_job_description(text: str) -> tuple[bool, str]:
    if not text.strip():
        return True, ""
    if len(text.strip()) < 30:
        return False, "⚠️ The job description is too short. Please paste a proper job posting."
    if len(text.strip()) > 5000:
        return False, "⚠️ The job description is too long. Please trim it to under 5000 characters."
    return True, ""


def check_cv_file(uploaded_file) -> tuple[bool, str]:
    """Validate an uploaded CV file for type and size."""
    if uploaded_file is None:
        return True, ""

    filename = uploaded_file.name.lower()
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"⚠️ Unsupported file type '{ext}'. "
            "Please upload a PDF (.pdf) or Word (.docx) file."
        )

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, (
            f"⚠️ File is too large ({size_mb:.1f} MB). "
            f"Please upload a file under {MAX_FILE_SIZE_MB} MB."
        )

    return True, ""


def check_content_quality(text: str) -> tuple[bool, str]:
    """
    Catch meaningless inputs using two simple character-based rules.
    Never checks topic or vocabulary — no false positives for domain terms.
    Skipped for answers over 20 words since effort was clearly made.

    Rule 1: >90% non-alphabetic — catches "123456", "!@#$%", "xkqz123!!"
    Rule 2: <8% vowels (in strings with 8+ alpha chars) — catches consonant-only
            mashing like "xkqzprtv". Does not catch all random strings that happen
            to contain vowels — the AI handles those by giving a very low score.
    """
    stripped = text.strip()
    MSG = (
        "⚠️ Your answer doesn't seem to contain meaningful text. "
        "Please write a proper response to the interview question."
    )

    # Always pass long answers through
    if len(stripped.split()) > 20:
        return True, ""

    total = len(stripped)
    if total == 0:
        return True, ""

    # Rule 1 — block if more than 90% of characters are non-alphabetic
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count / total < 0.10:
        return False, MSG

    # Rule 2 — block if vowel ratio is below 8% for longer alphabetic strings
    # Only applied when there are enough chars to make a reliable judgement
    if alpha_count >= 8:
        vowel_count = sum(1 for c in stripped.lower() if c in "aeiou")
        if vowel_count / alpha_count < 0.08:
            return False, MSG

    return True, ""


def validate_input(text: str) -> tuple[bool, str]:
    """
    Run security guards on a user answer. Returns (True, '') if safe.
    Applies: length check, injection detection, content quality check.
    """
    for check in [check_length, check_injection, check_content_quality]:
        is_safe, reason = check(text)
        if not is_safe:
            return False, reason
    return True, ""
