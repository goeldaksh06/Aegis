import pytest

from app.llm.base import BaseLLMProvider, LLMResponse
from app.llm.factory import get_provider, register_provider, supported_providers


class StubProvider(BaseLLMProvider):
    async def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        return LLMResponse(content=prompt, model=model, provider="stub")


def test_supported_providers_exposes_registry_keys():
    providers = supported_providers()

    assert "gemini" in providers
    assert "openai" in providers


def test_register_provider_adds_new_provider_without_factory_changes():
    register_provider("stub", StubProvider)

    provider = get_provider(" stub ")

    assert isinstance(provider, StubProvider)


def test_get_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_provider("unknown")
