import pytest

from app.agents.planner_agent import PlannerAgent
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
                    content="Milestone 1: scope the project. Milestone 2: build the prototype.",
                    model="stub-model",
                    provider="stub-provider",
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=2.5,
                ),
                "routing": RoutingResult(
                    agent=AgentType.PLANNER,
                    model="stub-model",
                    provider="stub-provider",
                    confidence=1.0,
                    reason="stub",
                ),
            },
        )()


@pytest.mark.asyncio
async def test_planner_agent_delegates_to_llm_service_and_builds_brief():
    service = StubLLMService()
    agent = PlannerAgent(llm_service=service)

    response = await agent.handle(
        ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="Plan the roadmap for this project")],
            preferred_provider="gemini",
        )
    )

    assert response.routing.agent == AgentType.PLANNER
    assert len(service.requests) == 1

    request, policy, system_prompt = service.requests[0]
    assert request.metadata["agent"] == "planner"
    assert policy.preferred_provider == "gemini"
    assert "Planner Agent" in system_prompt
    assert response.mission_brief is not None
