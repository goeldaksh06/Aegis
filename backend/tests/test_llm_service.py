import pytest

from app.llm.base import BaseLLMProvider, LLMResponse
from app.llm.registry import MODEL_REGISTRY, ModelConfig, register_model
from app.llm.service import LLMService
from app.models.schemas import ProviderCallRequest
from app.routing.model_router import ModelRouter
from app.routing.policies import RoutingPolicy


class StubProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    async def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        self.calls.append((prompt, model, system_prompt))
        return LLMResponse(
            content="stubbed response",
            model=model,
            provider="stub",
            input_tokens=3,
            output_tokens=2,
            latency_ms=1.5,
        )


@pytest.fixture(autouse=True)
def setup_registry():
    MODEL_REGISTRY.clear()

    register_model(
        ModelConfig(
            name="stub-model",
            provider="stub",
            tier="balanced",
            supports_tools=True,
            supports_vision=False,
            quality_score=0.8,
            speed_score=0.7,
            cost_score=0.6,
        )
    )

    yield

    MODEL_REGISTRY.clear()


@pytest.mark.asyncio
async def test_llm_service_routes_and_normalizes_response():
    provider = StubProvider()

    def provider_factory(provider_name: str) -> BaseLLMProvider:
        assert provider_name == "stub"
        return provider

    service = LLMService(
        router=ModelRouter(),
        provider_factory=provider_factory,
    )

    result = await service.generate(
        ProviderCallRequest(
            prompt="hello",
            model="ignored-by-service",
            metadata={"agent": "research"},
        ),
        RoutingPolicy(preference="speed"),
        system_prompt="be precise",
    )

    assert result.response.content == "stubbed response"
    assert result.response.model == "stub-model"
    assert result.response.provider == "stub"
    assert result.routing.model == "stub-model"
    assert result.routing.provider == "stub"
