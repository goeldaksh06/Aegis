from fastapi.testclient import TestClient

from app.api.chat import router as chat_router
from app.api.chat_service import ChatService
from app.app_container import get_chat_service
from app.main import app
from app.models.schemas import (
    AgentType,
    ChatResponse,
    ObservabilityEvent,
    ObservabilityEventType,
    RoutingResult,
)


class StubChatService:
    async def chat(self, request, *, user_id=None):
        return type(
            "Result",
            (),
            {
                "response": ChatResponse(
                    content="chat api answer",
                    routing=RoutingResult(
                        agent=AgentType.RESEARCH,
                        model="stub-model",
                        provider="stub-provider",
                        confidence=1.0,
                        reason="stub",
                    ),
                    telemetry=ObservabilityEvent(
                        event_type=ObservabilityEventType.REQUEST_COMPLETED,
                        request_id="req-1",
                    ),
                )
            },
        )()


def test_chat_endpoint_returns_chat_response():
    app.dependency_overrides[get_chat_service] = lambda: StubChatService()
    app.include_router(chat_router)

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Research the market"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == "chat api answer"
    assert response.json()["routing"]["agent"] == "research"

    app.dependency_overrides.clear()
