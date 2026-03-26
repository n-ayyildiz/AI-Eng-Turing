# pricing.py
# Estimates the cost of each OpenAI API call based on token usage.
# Prices are per 1,000,000 tokens (as published by OpenAI, April 2025).

# ----------------------------------------------------------------------
# MODEL PRICING TABLE (USD per 1M tokens)
# ----------------------------------------------------------------------
MODEL_PRICING = {
    "gpt-4.1":       {"input": 2.00,  "output": 8.00},
    "gpt-4.1-mini":  {"input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":  {"input": 0.10,  "output": 0.40},
    "gpt-4o":        {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.60},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> dict:
    """
    Calculate the USD cost for a single API call.

    Returns a dict with:
      - input_tokens
      - output_tokens
      - total_tokens
      - input_cost_usd
      - output_cost_usd
      - total_cost_usd
      - formatted: a human-readable string
    """
    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "formatted": (
            f"🪙 **This call:** {input_tokens + output_tokens:,} tokens "
            f"(${total_cost:.6f})"
        ),
    }


def format_session_cost(calls: list[dict]) -> str:
    """
    Summarise cost across all API calls in a session.
    `calls` is a list of dicts returned by calculate_cost().
    """
    if not calls:
        return "No API calls made yet."

    total_tokens = sum(c["total_tokens"] for c in calls)
    total_cost = sum(c["total_cost_usd"] for c in calls)
    num_calls = len(calls)

    return (
        f"📊 **Session total:** {num_calls} call(s) | "
        f"{total_tokens:,} tokens | "
        f"${total_cost:.6f} USD"
    )
