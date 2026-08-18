import pytest

from app.agents.coder_agent import CoderAgent
from app.models.schemas import (
    AgentType,
    ChatMessage,
    ChatRequest,
    MessageRole,
    ProviderCallResult,
    RoutingResult,
)


class StubLLMService:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request, policy, *, system_prompt=None):
        self.requests.append((request, policy, system_prompt))
        return type(
            "Execution",
            (),
            {
                "response": ProviderCallResult(
                    content="Recommend refactoring the function to fix the null pointer bug.",
                    model="stub-model",
                    provider="stub-provider",
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=2.5,
                ),
                "routing": RoutingResult(
                    agent=AgentType.CODER,
                    model="stub-model",
                    provider="stub-provider",
                    confidence=1.0,
                    reason="stub",
                ),
            },
        )()


@pytest.mark.asyncio
async def test_coder_agent_delegates_to_llm_service_and_builds_brief():
    service = StubLLMService()
    agent = CoderAgent(llm_service=service)

    response = await agent.handle(
        ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="Debug this function")],
            preferred_provider="gemini",
        )
    )

    assert response.routing.agent == AgentType.CODER
    assert len(service.requests) == 1

    request, policy, system_prompt = service.requests[0]
    assert request.metadata["agent"] == "coder"
    assert policy.preferred_provider == "gemini"
    assert "Coder Agent" in system_prompt
    assert response.mission_brief is not None
