import pytest

from app.events.bus import EventBus
from app.agents.router import AgentRouter
from app.api.chat_service import ChatResult, ChatService
from app.models.schemas import (
    AgentType,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MessageRole,
    RoutingResult,
)


class StubAgent:
    def __init__(self) -> None:
        self.calls = []

    async def handle(self, request: ChatRequest, *, on_stage=None) -> ChatResponse:
        self.calls.append(request)
        return ChatResponse(
            content="chat answer",
            routing=RoutingResult(
                agent=AgentType.RESEARCH,
                model="stub-model",
                provider="stub-provider",
                confidence=1.0,
                reason="stub",
            ),
        )


class RecordingEventBus(EventBus):
    def __init__(self) -> None:
        super().__init__()
        self.published = []

    async def publish(self, event):
        self.published.append(event)
        await super().publish(event)


def test_chat_service_requires_agents():
    with pytest.raises(ValueError):
        ChatService(agent_router=AgentRouter(), agents={})


@pytest.mark.asyncio
async def test_chat_service_routes_and_executes_selected_agent():
    research_agent = StubAgent()
    event_bus = RecordingEventBus()
    service = ChatService(
        agent_router=AgentRouter(),
        agents={AgentType.RESEARCH: research_agent},
        event_bus=event_bus,
    )

    result = await service.chat(
        ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="Research the market")],
        )
    )

    assert isinstance(result, ChatResult)
    assert result.response.content == "chat answer"
    assert len(research_agent.calls) == 1
    assert [event.event_type.value for event in event_bus.published] == [
        "request.received",
        "agent.selected",
        "request.completed",
    ]
