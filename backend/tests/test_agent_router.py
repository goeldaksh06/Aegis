from app.agents.router import agent_router
from app.models.schemas import AgentRoutingInput, AgentType


def test_agent_router_prefers_explicit_agent():
    decision = agent_router.route(
        AgentRoutingInput(
            task="research the market",
            preferred_agent=AgentType.PLANNER,
        )
    )

    assert decision.agent == AgentType.PLANNER
    assert decision.confidence == 1.0


def test_agent_router_routes_code_tasks_to_coder():
    decision = agent_router.route(
        AgentRoutingInput(task="implement a new service and debug it")
    )

    assert decision.agent == AgentType.CODER
    assert decision.confidence == 0.82


def test_agent_router_defaults_to_research():
    decision = agent_router.route(
        AgentRoutingInput(task="help me understand this problem")
    )

    assert decision.agent == AgentType.RESEARCH
    assert decision.confidence == 0.5
