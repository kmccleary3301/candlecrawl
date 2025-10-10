from __future__ import annotations

from typing import Tuple


# Per-million token pricing (USD): (input_per_million, output_per_million)
# These are reasonable defaults based on public guidance and internal usage; adjust as needed.
MODEL_PRICE_PER_MILLION = {
    # OpenAI/Router family (approximate)
    "openai/gpt-5-nano": (0.15, 0.60),
    "openai/gpt-5-mini": (0.30, 1.20),
    "openai/gpt-4.1-mini": (0.60, 2.40),
    "openai/o4-mini": (1.10, 4.40),
    "openai/o3-mini": (0.80, 3.20),

    # Perplexity (deep research)
    "perplexity/sonar-deep-research": (2.00, 8.00),  # + web search fees if applicable
}


def get_model_rates(model_name: str) -> Tuple[float, float]:
    key = (model_name or "").strip().lower()
    # Normalize common prefixes
    rates = None
    if key in MODEL_PRICE_PER_MILLION:
        rates = MODEL_PRICE_PER_MILLION[key]
    else:
        # Try to match by suffix (e.g., model card id without provider)
        for m, r in MODEL_PRICE_PER_MILLION.items():
            if key.endswith(m.lower()):
                rates = r
                break
    # Conservative fallback: $1.00 in / $4.00 out per million tokens
    return rates or (1.00, 4.00)


def estimate_cost_usd(model_name: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
    in_rate, out_rate = get_model_rates(model_name)
    return (prompt_tokens * (in_rate / 1_000_000.0)) + (completion_tokens * (out_rate / 1_000_000.0))

