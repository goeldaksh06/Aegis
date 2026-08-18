from dataclasses import dataclass
from typing import Literal


ModelTier = Literal["fast", "balanced", "powerful"]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    tier: ModelTier

    supports_tools: bool = False
    supports_vision: bool = False

    # Relative routing values for now.
    # We will replace/augment these with measured benchmarks later.
    quality_score: float = 0.5
    speed_score: float = 0.5
    cost_score: float = 0.5


MODEL_REGISTRY: dict[str, ModelConfig] = {}


def register_model(config: ModelConfig) -> None:
    """Register a model with Aegis."""
    if config.name in MODEL_REGISTRY:
        raise ValueError(f"Model '{config.name}' is already registered.")

    MODEL_REGISTRY[config.name] = config


def get_model(name: str) -> ModelConfig:
    """Return configuration for a registered model."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: '{name}'") from exc


def get_models_by_tier(tier: ModelTier) -> list[ModelConfig]:
    """Return all models belonging to a routing tier."""
    return [
        model
        for model in MODEL_REGISTRY.values()
        if model.tier == tier
    ]


def get_models_by_provider(provider: str) -> list[ModelConfig]:
    """Return all models belonging to a provider."""
    return [
        model
        for model in MODEL_REGISTRY.values()
        if model.provider == provider
    ]