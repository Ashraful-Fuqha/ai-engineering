"""
Versioned pricing catalogs for LLM models (Prices in USD per 1 Million tokens).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


# Pricing rates (as of Google Gemini AI Studio rate card)
PRICING_CATALOG: dict[str, ModelPrice] = {
    # Gemini 2.5 Flash
    "gemini-2.5-flash": ModelPrice(
        input_per_million=0.075,
        cached_input_per_million=0.01875,
        output_per_million=0.30,
    ),
    # Gemini 2.0 Flash / Flash-Lite
    "gemini-2.0-flash": ModelPrice(
        input_per_million=0.10,
        cached_input_per_million=0.025,
        output_per_million=0.40,
    ),
    "gemini-2.0-flash-lite": ModelPrice(
        input_per_million=0.075,
        cached_input_per_million=0.01875,
        output_per_million=0.30,
    ),
    # Gemini 1.5 Flash (legacy reference)
    "gemini-1.5-flash": ModelPrice(
        input_per_million=0.075,
        cached_input_per_million=0.01875,
        output_per_million=0.30,
    ),
    # Gemini 1.5 Pro
    "gemini-1.5-pro": ModelPrice(
        input_per_million=1.25,
        cached_input_per_million=0.3125,
        output_per_million=5.00,
    ),
    # Default fallback rate for other models (e.g. gpt-4o-mini equivalent)
    "default": ModelPrice(
        input_per_million=0.15,
        cached_input_per_million=0.075,
        output_per_million=0.60,
    ),
}


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Calculate estimated cost in USD based on token counts and model pricing."""
    rate = PRICING_CATALOG.get(model.lower(), PRICING_CATALOG["default"])
    
    # Non-cached prompt tokens
    uncached_prompt_tokens = max(0, prompt_tokens - cached_tokens)
    
    input_cost = (uncached_prompt_tokens / 1_000_000) * rate.input_per_million
    cached_cost = (cached_tokens / 1_000_000) * rate.cached_input_per_million
    output_cost = (completion_tokens / 1_000_000) * rate.output_per_million
    
    return round(input_cost + cached_cost + output_cost, 8)
