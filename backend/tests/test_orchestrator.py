import pytest

from app.agents.orchestrator import run_orchestrated_plan
from app.agents.router import agent_router
from app.models.schemas import (
    AgentResult,
    AgentType,
    ChatRequest,
    ChatMessage,
    CostEstimate,
    MessageRole,
    MissionBrief,
    RiskLevel,
    RoutingResult,
)


def _routing(agent: AgentType) -> RoutingResult:
    return RoutingResult(agent=agent, model="stub-model", provider="stub", confidence=1.0, reason="stub")


def _cost() -> CostEstimate:
    return CostEstimate(input_tokens=10, output_tokens=10, cost_usd=0.001, model="stub-model", provider="stub")


class StubPlannerAgent:
    """Returns a plan whose recommended_actions map onto research/coder/document keywords."""

    async def handle(self, request, *, on_stage=None):
        return AgentResult(
            content="Here is the plan.",
            routing=_routing(AgentType.PLANNER),
            mission_brief=MissionBrief(
                summary="plan summary",
                risk_score=80,
                risk_level=RiskLevel.HIGH,
                top_alerts=["alert"],
                recommended_actions=[
                    "Research the root cause of the outage",
                    "Debug and fix the failing service",
                ],
                evidence=["evidence"],
            ),
            cost=_cost(),
        )


class StubSubAgent:
    def __init__(self, agent_type: AgentType) -> None:
        self.agent_type = agent_type
        self.received_requests: list[ChatRequest] = []

    async def handle(self, request, *, on_stage=None):
        self.received_requests.append(request)
        return AgentResult(
            content=f"handled by {self.agent_type.value}",
            routing=_routing(self.agent_type),
            mission_brief=MissionBrief(
                summary="sub summary",
                risk_score=50,
                risk_level=RiskLevel.MEDIUM,
                top_alerts=["sub alert"],
                recommended_actions=["sub action"],
                evidence=["sub evidence"],
            ),
            cost=_cost(),
        )


@pytest.mark.asyncio
async def test_orchestrator_dispatches_sub_steps_to_different_agents():
    research_agent = StubSubAgent(AgentType.RESEARCH)
    coder_agent = StubSubAgent(AgentType.CODER)

    combined, sub_results = await run_orchestrated_plan(
        planner_agent=StubPlannerAgent(),
        agents={AgentType.RESEARCH: research_agent, AgentType.CODER: coder_agent},
        agent_router=agent_router,
        request=ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="Plan the incident response")]
        ),
    )

    assert len(sub_results) == 2
    assert {r.routing.agent for r in sub_results} == {AgentType.RESEARCH, AgentType.CODER}
    assert len(research_agent.received_requests) == 1
    assert len(coder_agent.received_requests) == 1

    assert "Here is the plan." in combined.content
    assert "handled by research" in combined.content
    assert "handled by coder" in combined.content
    assert combined.cost is not None
    # combined cost = planner + 2 sub-agents
    assert combined.cost.cost_usd == pytest.approx(0.003, abs=1e-9)


@pytest.mark.asyncio
async def test_orchestrator_returns_plan_only_when_no_mission_brief():
    class BareAgent:
        async def handle(self, request, *, on_stage=None):
            return AgentResult(content="just text", routing=_routing(AgentType.PLANNER))

    combined, sub_results = await run_orchestrated_plan(
        planner_agent=BareAgent(),
        agents={},
        agent_router=agent_router,
        request=ChatRequest(messages=[ChatMessage(role=MessageRole.USER, content="Plan something")]),
    )

    assert sub_results == []
    assert combined.content == "just text"
