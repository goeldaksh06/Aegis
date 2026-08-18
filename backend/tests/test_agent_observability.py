from fastapi.testclient import TestClient

from app.agents.router import agent_router
from app.api.chat_service import ChatService
from app.app_container import get_chat_service
from app.database import db
from app.main import app
from app.models.schemas import (
    AgentResult,
    AgentType,
    MissionBrief,
    ResponseEvaluation,
    RiskLevel,
    RoutingResult,
    CostEstimate,
)


def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_observability.db")
    monkeypatch.setattr(db, "DATABASE_URL", f"sqlite+aiosqlite:///{db.DB_PATH}")
    db.get_engine.cache_clear()
    db.get_sessionmaker.cache_clear()
    db._ready = False


class ObservableStubAgent:
    """Emits a real 'generation' on_stage event, like execute_agent_turn does."""

    def __init__(self, agent_type: AgentType, model: str = "mock-default") -> None:
        self.agent_type = agent_type
        self.model = model

    async def handle(self, request, *, on_stage=None):
        if on_stage is not None:
            await on_stage(
                "generation",
                {
                    "agent": self.agent_type.value,
                    "provider": "mock",
                    "model": self.model,
                    "elapsed_ms": 42.0,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.0021,
                    "retrieved_count": 3,
                },
            )
        return AgentResult(
            content=f"answer from {self.agent_type.value}",
            routing=RoutingResult(
                agent=self.agent_type, model=self.model, provider="mock", confidence=1.0, reason="stub"
            ),
            mission_brief=MissionBrief(
                summary="s", risk_score=50, risk_level=RiskLevel.MEDIUM,
                top_alerts=["a"], recommended_actions=["b"], evidence=["c"],
            ),
            evaluation=ResponseEvaluation(structure_quality=1.0, overall_score=1.0),
            cost=CostEstimate(input_tokens=100, output_tokens=50, cost_usd=0.0021, model=self.model, provider="mock"),
        )


def test_chat_endpoint_persists_agent_steps_queryable_via_run_detail(tmp_path, monkeypatch):
    _isolated_db(tmp_path, monkeypatch)

    service = ChatService(
        agent_router=agent_router,
        agents={AgentType.RESEARCH: ObservableStubAgent(AgentType.RESEARCH)},
    )
    app.dependency_overrides[get_chat_service] = lambda: service

    client = TestClient(app)
    register = client.post("/auth/register", json={"email": "obs@example.com", "password": "correct-horse-1"})
    token = register.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    chat_response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Research something"}]},
        headers=headers,
    )
    assert chat_response.status_code == 200
    run_id = chat_response.json()["run_id"]
    assert run_id

    detail_response = client.get(f"/runs/{run_id}", headers=headers)
    assert detail_response.status_code == 200
    detail = detail_response.json()

    assert len(detail["steps"]) == 1
    step = detail["steps"][0]
    assert step["agent"] == "research"
    assert step["duration_ms"] == 42.0
    assert step["input_tokens"] == 100
    assert step["output_tokens"] == 50
    assert step["cost_usd"] == 0.0021
    assert detail["total_tokens"] == 150
    assert detail["total_cost_usd"] == 0.0021

    app.dependency_overrides.clear()
