from __future__ import annotations

from collections.abc import Callable

from app.llm.base import BaseLLMProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.openai import OpenAIProvider
from app.llm.providers.mock_provider import MockProvider


ProviderConstructor = Callable[[], BaseLLMProvider]

_PROVIDER_REGISTRY: dict[str, ProviderConstructor] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "mock": MockProvider,
}


def register_provider(name: str, constructor: ProviderConstructor) -> None:
    normalized = name.lower().strip()

    if not normalized:
        raise ValueError("Provider name cannot be empty.")

    _PROVIDER_REGISTRY[normalized] = constructor


def supported_providers() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDER_REGISTRY))


def get_provider(provider: str) -> BaseLLMProvider:
    normalized = provider.lower().strip()

    constructor = _PROVIDER_REGISTRY.get(normalized)

    if constructor is not None:
        return constructor()

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'"
    )