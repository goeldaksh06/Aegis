from app.llm.registry import (
    MODEL_REGISTRY,
    ModelConfig,
    register_model,
)
from app.config.settings import settings


def register_default_models() -> None:
    if MODEL_REGISTRY:
        return

    register_model(
        ModelConfig(
            name=settings.MODEL_NAME,
            provider=settings.MODEL_PROVIDER,
            tier="balanced",
            supports_tools=True,
            supports_vision=True,
            quality_score=0.80,
            speed_score=0.80,
            cost_score=0.80,
        )
    )

    if settings.MODEL_NAME != "gemini-2.5-flash-lite":
        register_model(
            ModelConfig(
                name="gemini-2.5-flash-lite",
                provider="gemini",
                tier="fast",
                supports_tools=True,
                supports_vision=True,
                quality_score=0.70,
                speed_score=0.95,
                cost_score=0.95,
            )
        )

    if settings.MODEL_NAME != "gemini-2.5-flash":
        register_model(
            ModelConfig(
                name="gemini-2.5-flash",
                provider="gemini",
                tier="balanced",
                supports_tools=True,
                supports_vision=True,
                quality_score=0.85,
                speed_score=0.85,
                cost_score=0.80,
            )
        )