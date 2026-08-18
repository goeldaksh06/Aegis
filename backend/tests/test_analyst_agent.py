import pytest

from app.agents.analyst_agent import AnalystAgent
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
                    content="The critical risk requires immediate action. We recommend escalation.",
                    model="stub-model",
                    provider="stub-provider",
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=2.5,
                ),
                "routing": RoutingResult(
                    agent=AgentType.ANALYST,
                    model="stub-model",
                    provider="stub-provider",
                    confidence=1.0,
                    reason="stub",
                ),
            },
        )()


@pytest.mark.asyncio
async def test_analyst_agent_delegates_to_llm_service_and_builds_brief():
    service = StubLLMService()
    agent = AnalystAgent(llm_service=service)

    response = await agent.handle(
        ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="Analyze the supply chain risk")],
            preferred_provider="gemini",
        )
    )

    assert response.routing.agent == AgentType.ANALYST
    assert len(service.requests) == 1

    request, policy, system_prompt = service.requests[0]
    assert request.metadata["agent"] == "analyst"
    assert policy.preferred_provider == "gemini"
    assert "Analyst Agent" in system_prompt
    assert response.mission_brief is not None
    assert response.mission_brief.risk_level in {"low", "medium", "high"}
