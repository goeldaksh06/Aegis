import pytest
from pydantic import ValidationError

from app.models.schemas import (
    AgentType,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MessageRole,
    ObservabilityEvent,
    ObservabilityEventType,
    RoutingResult,
)


def test_chat_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="Hello")],
            unexpected="value",
        )


def test_chat_response_envelope_shape():
    response = ChatResponse(
        content="ok",
        routing=RoutingResult(
            agent=AgentType.RESEARCH,
            model="gemini-2.5-flash",
            provider="gemini",
            confidence=0.9,
            reason="high confidence",
        ),
        telemetry=ObservabilityEvent(
            event_type=ObservabilityEventType.REQUEST_COMPLETED,
            request_id="req_123",
        ),
    )

    assert response.routing.model == "gemini-2.5-flash"
    assert response.routing.agent == AgentType.RESEARCH
    assert response.telemetry is not None
