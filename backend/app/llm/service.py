from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.llm.base import BaseLLMProvider, LLMResponse
from app.llm.factory import get_provider
from app.models.schemas import ProviderCallRequest, ProviderCallResult, RoutingResult
from app.routing.model_router import ModelRouter, model_router
from app.routing.policies import RoutingPolicy


ProviderFactory = Callable[[str], BaseLLMProvider]


@dataclass(frozen=True)
class LLMExecution:
    response: ProviderCallResult
    routing: RoutingResult


class LLMService:
    def __init__(
        self,
        router: ModelRouter | None = None,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self._router = router or model_router
        self._provider_factory = provider_factory or get_provider

    async def generate(
        self,
        request: ProviderCallRequest,
        policy: RoutingPolicy,
        *,
        system_prompt: str | None = None,
    ) -> LLMExecution:
        selected_model = self._router.route(policy)
        provider = self._provider_factory(selected_model.provider)

        normalized_system_prompt = system_prompt or request.system_prompt

        response = await provider.generate(
            prompt=request.prompt,
            model=selected_model.name,
            system_prompt=normalized_system_prompt,
        )

        routing = RoutingResult(
            agent=request.metadata.get("agent", "research"),
            model=selected_model.name,
            provider=selected_model.provider,
            preference=policy.preference,
            confidence=1.0,
            reason="selected by model router",
            metadata={
                "requested_model": request.model,
                "provider": request.provider,
                **request.metadata,
            },
        )

        return LLMExecution(
            response=self._to_result(response),
            routing=routing,
        )

    @staticmethod
    def _to_result(response: LLMResponse) -> ProviderCallResult:
        return ProviderCallResult(
            content=response.content,
            model=response.model,
            provider=response.provider,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )
