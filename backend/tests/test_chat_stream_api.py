from fastapi.testclient import TestClient

from app.agents.router import agent_router
from app.api.chat_stream import router as chat_stream_router
from app.app_container import get_chat_service
from app.main import app
from app.models.schemas import (
    AgentResult,
    AgentType,
    MissionBrief,
    ResponseEvaluation,
    RiskLevel,
    RoutingResult,
)


class StubAgent:
    async def handle(self, request, *, on_stage=None):
        if on_stage is not None:
            await on_stage("retrieval", {"retrieved_count": 0, "sources": [], "elapsed_ms": 1.0})
            await on_stage(
                "generation",
                {"provider": "mock", "model": "mock-default", "elapsed_ms": 2.0, "output_tokens": 5},
            )
            await on_stage("brief", {"risk_score": 42, "risk_level": "medium"})
            await on_stage("evaluated", {"overall_score": 0.8, "groundedness": None, "structure_quality": 1.0})

        return AgentResult(
            content="streamed answer",
            routing=RoutingResult(
                agent=AgentType.RESEARCH,
                model="mock-default",
                provider="mock",
                confidence=1.0,
                reason="stub",
            ),
            mission_brief=MissionBrief(
                summary="summary",
                risk_score=42,
                risk_level=RiskLevel.MEDIUM,
                top_alerts=["alert"],
                recommended_actions=["action"],
                evidence=["evidence"],
            ),
            evaluation=ResponseEvaluation(
                groundedness=None, structure_quality=1.0, overall_score=0.8
            ),
        )


class StubChatService:
    agent_router = agent_router
    agents = {AgentType.RESEARCH: StubAgent()}


def _parse_sse(body: str) -> list[tuple[str, str]]:
    events = []
    for chunk in body.strip().split("\n\n"):
        lines = chunk.split("\n")
        event = next(line.split("event: ")[1] for line in lines if line.startswith("event: "))
        data = next(line.split("data: ")[1] for line in lines if line.startswith("data: "))
        events.append((event, data))
    return events


def test_chat_stream_emits_every_stage_and_final_response():
    app.dependency_overrides[get_chat_service] = lambda: StubChatService()
    app.include_router(chat_stream_router)

    client = TestClient(app)
    with client.stream(
        "POST",
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "research something"}]},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    stage_names = [name for name, _ in events]

    assert stage_names == [
        "received",
        "routed",
        "retrieval",
        "generation",
        "brief",
        "evaluated",
        "persisted",
        "done",
    ]

    import json

    done_payload = json.loads(dict(events)["done"])
    assert done_payload["content"] == "streamed answer"
    assert done_payload["routing"]["agent"] == "research"
    assert done_payload["mission_brief"]["risk_score"] == 42

    app.dependency_overrides.clear()


def test_chat_stream_emits_error_event_for_unregistered_agent():
    class EmptyChatService:
        agent_router = agent_router
        agents: dict = {}

    app.dependency_overrides[get_chat_service] = lambda: EmptyChatService()
    app.include_router(chat_stream_router)

    client = TestClient(app)
    with client.stream(
        "POST",
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "research something"}]},
    ) as response:
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    assert events[-1][0] == "error"

    app.dependency_overrides.clear()
