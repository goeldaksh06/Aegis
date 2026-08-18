from __future__ import annotations

from dataclasses import dataclass

# Approximate list prices in USD per 1,000 tokens, as of when this project was built. These
# are for cost *estimation* in the telemetry/demo UI, not billing-accurate accounting — real
# provider invoices are the source of truth for actual spend. Unknown models fall back to
# _DEFAULT_PRICE rather than raising, since the model registry is extensible (app/llm/
# default_models.py) and pricing shouldn't be a hard dependency for a new model to work.


@dataclass(frozen=True)
class ModelPrice:
    input_per_1k: float
    output_per_1k: float


_PRICING: dict[str, ModelPrice] = {
    "mock-default": ModelPrice(input_per_1k=0.0, output_per_1k=0.0),
    "deepseek-chat": ModelPrice(input_per_1k=0.00027, output_per_1k=0.0011),
    "deepseek-v4-pro": ModelPrice(input_per_1k=0.00055, output_per_1k=0.00219),
    "gemini-2.5-flash": ModelPrice(input_per_1k=0.0003, output_per_1k=0.0025),
    "gemini-2.5-flash-lite": ModelPrice(input_per_1k=0.0001, output_per_1k=0.0004),
}

_DEFAULT_PRICE = ModelPrice(input_per_1k=0.001, output_per_1k=0.003)


def estimate_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float:
    price = _PRICING.get(model, _DEFAULT_PRICE)
    input_cost = ((input_tokens or 0) / 1000) * price.input_per_1k
    output_cost = ((output_tokens or 0) / 1000) * price.output_per_1k
    return round(input_cost + output_cost, 6)
