import pytest

from app.agents.document_agent import DocumentAgent
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
                    content="The contract clause requires 30 days notice before termination.",
                    model="stub-model",
                    provider="stub-provider",
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=2.5,
                ),
                "routing": RoutingResult(
                    agent=AgentType.DOCUMENT,
                    model="stub-model",
                    provider="stub-provider",
                    confidence=1.0,
                    reason="stub",
                ),
            },
        )()


@pytest.mark.asyncio
async def test_document_agent_delegates_to_llm_service_and_builds_brief():
    service = StubLLMService()
    agent = DocumentAgent(llm_service=service)

    response = await agent.handle(
        ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="Summarize this contract")],
            preferred_provider="gemini",
        )
    )

    assert response.routing.agent == AgentType.DOCUMENT
    assert len(service.requests) == 1

    request, policy, system_prompt = service.requests[0]
    assert request.metadata["agent"] == "document"
    assert policy.preferred_provider == "gemini"
    assert "Document Agent" in system_prompt
    assert response.mission_brief is not None
