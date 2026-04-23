"""
Query validation and safety guard.

Validates user queries before running the pipeline. Catches:
- Gibberish / random alphanumeric strings
- Off-topic queries unrelated to science/biomedical research
- Prompt injection attempts
- Unethical or harmful queries

System role (not shown to user):
    This tool is for creating research hypotheses from biomedical and
    neuroscience literature. It processes scientific queries only and
    should not respond to off-topic, harmful, or adversarial inputs.

Public API:
    - validate_query(prompt: str) -> tuple[bool, str]
      Returns (is_valid, warning_message). If is_valid is True,
      warning_message is empty.
"""

from __future__ import annotations

import re

# =============================================================================
# System role (internal — not displayed to user)
# =============================================================================

SYSTEM_ROLE = (
    "This tool is for creating research hypotheses from scientific literature. "
    "It is intended exclusively for biomedical and neuroscience research use. "
    "Queries must be scientific, ethical, and relevant to research topics."
)

# =============================================================================
# Pattern definitions
# =============================================================================

# Prompt injection — attempts to override system behaviour
_INJECTION_PATTERNS = re.compile(
    r"(?i)\b("
    r"ignore\s+(previous|all|prior)\s+instructions?"
    r"|disregard\s+(previous|all|prior)\s+instructions?"
    r"|you\s+are\s+now\s+a"
    r"|act\s+as\s+(a\s+|an\s+)?(different|new|unrestricted)"
    r"|pretend\s+(you\s+are|to\s+be)"
    r"|forget\s+(your|all)\s+(instructions?|rules?|guidelines?)"
    r"|override\s+(your\s+)?(instructions?|rules?|guidelines?)"
    r"|new\s+persona"
    r"|jailbreak"
    r"|do\s+anything\s+now"
    r"|dan\s+mode"
    r")\b"
)

# Ethical / harmful intent
_HARMFUL_PATTERNS = re.compile(
    r"(?i)\b("
    r"how\s+to\s+(make|create|build|synthesize)\s+(a\s+)?(bomb|weapon|poison|drug|virus|malware)"
    r"|suicide\s+method"
    r"|self.harm"
    r"|kill\s+(a\s+)?(person|human|people)"
    r"|bioweapon"
    r"|chemical\s+weapon"
    r"|nerve\s+agent"
    r")\b"
)

# Clearly off-topic — non-scientific everyday topics
_OFF_TOPIC_PATTERNS = re.compile(
    r"(?i)^("
    r"what('?s|\s+is)\s+the\s+(weather|time|date|score)"
    r"|who\s+(won|is\s+winning)"
    r"|recipe\s+for"
    r"|how\s+to\s+cook"
    r"|best\s+(restaurant|movie|song|game|phone)"
    r"|sports?\s+(score|result)"
    r"|stock\s+price"
    r"|lottery"
    r"|celebrity"
    r")\b"
)

# Gibberish detection — random alphanumeric strings with no real words.
# Flags queries where every token is either:
#   (a) purely numeric, or
#   (b) a long (8+ char) string with no vowels, or
#   (c) a mix of letters+digits in a single token (e.g. "abc123xyz")
_GIBBERISH_TOKEN = re.compile(
    r"^("
    r"\d+"                          # purely numeric
    r"|[^aeiou\s]{8,}"             # 8+ consonants, no vowels
    r"|[a-z]*\d+[a-z]+\d*"        # letters-digits mix
    r"|[a-z]{1,2}\d{3,}"          # tiny letters + many digits
    r")$",
    re.IGNORECASE,
)


def _is_gibberish(text: str) -> bool:
    """
    Return True if the query appears to be random characters.

    Uses a token-level check: if ALL tokens match the gibberish pattern,
    the query is flagged. A single recognisable word saves it.
    """
    tokens = text.split()
    if not tokens:
        return False
    gibberish_count = sum(1 for t in tokens if _GIBBERISH_TOKEN.match(t))
    return gibberish_count == len(tokens)


# =============================================================================
# Main validation function
# =============================================================================

def validate_query(prompt: str) -> tuple[bool, str]:
    """
    Validate a user query before running the pipeline.

    Checks (in order):
        1. Prompt injection attempt
        2. Harmful / unethical content
        3. Gibberish / random string
        4. Clearly off-topic (non-scientific)

    Args:
        prompt: the raw user query string.

    Returns:
        (True, "") if the query is valid.
        (False, warning_message) if the query should be blocked.
    """
    text = prompt.strip()

    # 1. Injection
    if _INJECTION_PATTERNS.search(text):
        return False, "This query cannot be processed."

    # 2. Harmful / ethical
    if _HARMFUL_PATTERNS.search(text):
        return False, "This app is for ethical research queries only."

    # 3. Gibberish
    if _is_gibberish(text):
        return False, "Please enter a meaningful research topic."

    # 4. Off-topic
    if _OFF_TOPIC_PATTERNS.search(text):
        return False, (
            "Please enter a neuroscience or biomedical research topic. "
            "This tool is for creating research hypotheses."
        )

    return True, ""
