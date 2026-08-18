from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol

from app.agents.orchestrator import run_orchestrated_plan
from app.agents.support import StageCallback
from app.database.db import save_agent_steps, save_run
from app.events.bus import EventBus, event_bus
from app.models.schemas import EventRecord, EventType
from app.models.schemas import AgentResult, AgentRoutingInput, ChatMessage, ChatRequest, ChatResponse, AgentType
from app.agents.router import AgentRouter, agent_router


class ChatAgent(Protocol):
    async def handle(self, request: ChatRequest, *, on_stage: StageCallback | None = None) -> AgentResult:
        ...


@dataclass(frozen=True)
class ChatResult:
    response: ChatResponse
    selected_agent: AgentType
    run_id: str


def agent_result_to_response(
    agent_result: AgentResult,
    sub_results: list[AgentResult] | None = None,
    *,
    run_id: str | None = None,
) -> ChatResponse:
    return ChatResponse(
        content=agent_result.content,
        routing=agent_result.routing,
        telemetry=agent_result.telemetry,
        mission_brief=agent_result.mission_brief,
        tool_results=agent_result.tool_results,
        evaluation=agent_result.evaluation,
        cost=agent_result.cost,
        moderation=agent_result.moderation,
        conversation_id=agent_result.conversation_id,
        sub_results=[agent_result_to_response(sub) for sub in (sub_results or [])],
        run_id=run_id,
    )


def _collect_generation_steps(events: list[dict[str, object]]) -> list[dict[str, object]]:
    """Reduce raw on_stage events down to one row per real LLM call ("generation" stage).

    This is what powers per-agent observability (GET /runs/{id}) — a single-agent request
    produces one row, an orchestrated request produces one row per agent actually dispatched.
    """
    return [
        {
            "agent": event.get("agent"),
            "model": event.get("model"),
            "provider": event.get("provider"),
            "duration_ms": event.get("elapsed_ms"),
            "input_tokens": event.get("input_tokens"),
            "output_tokens": event.get("output_tokens"),
            "cost_usd": event.get("cost_usd"),
            "retrieved_count": event.get("retrieved_count"),
        }
        for event in events
    ]


@dataclass
class ChatService:
    agent_router: AgentRouter = agent_router
    agents: dict[AgentType, ChatAgent] = field(default_factory=dict)
    event_bus: EventBus = field(default_factory=lambda: event_bus)

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("ChatService requires at least one agent implementation.")

    async def chat(self, request: ChatRequest, *, user_id: str | None = None) -> ChatResult:
        await self.event_bus.publish(
            EventRecord(
                event_type=EventType.REQUEST_RECEIVED,
                metadata={"message_count": len(request.messages)},
            )
        )

        agent_input = self._build_agent_input(request)
        decision = self.agent_router.route(agent_input)
        await self.event_bus.publish(
            EventRecord(
                event_type=EventType.AGENT_SELECTED,
                agent=decision.agent,
                metadata={
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                },
            )
        )
        agent = self.agents.get(decision.agent)

        if agent is None:
            raise RuntimeError(f"No agent registered for '{decision.agent}'")

        run_id = str(uuid.uuid4())
        generation_events: list[dict[str, object]] = []

        async def on_stage(name: str, data: dict[str, object]) -> None:
            if name == "generation":
                generation_events.append(data)

        try:
            sub_results: list[AgentResult] = []
            if request.orchestrate:
                planner = self.agents.get(AgentType.PLANNER, agent)
                agent_result, sub_results = await run_orchestrated_plan(
                    planner_agent=planner,
                    agents=self.agents,
                    agent_router=self.agent_router,
                    request=request,
                    on_stage=on_stage,
                )
            else:
                agent_result = await agent.handle(request, on_stage=on_stage)

            response = agent_result_to_response(agent_result, sub_results, run_id=run_id)
            await self.event_bus.publish(
                EventRecord(
                    event_type=EventType.REQUEST_COMPLETED,
                    agent=decision.agent,
                    model=agent_result.routing.model,
                    provider=agent_result.routing.provider,
                    metadata={
                        "selected_agent": decision.agent.value,
                        "evaluation": agent_result.evaluation.model_dump()
                        if agent_result.evaluation
                        else None,
                        "cost": agent_result.cost.model_dump() if agent_result.cost else None,
                        "moderation": agent_result.moderation.model_dump()
                        if agent_result.moderation
                        else None,
                    },
                )
            )
            await save_run(
                id=run_id,
                user_id=user_id,
                prompt=agent_input.task,
                status="success",
                agent=decision.agent.value,
                model=agent_result.routing.model,
                provider=agent_result.routing.provider,
                risk_level=agent_result.mission_brief.risk_level.value
                if agent_result.mission_brief
                else None,
                risk_score=agent_result.mission_brief.risk_score
                if agent_result.mission_brief
                else None,
                cost_usd=agent_result.cost.cost_usd if agent_result.cost else None,
                moderation_blocked=agent_result.moderation.blocked if agent_result.moderation else None,
                conversation_id=agent_result.conversation_id,
            )
            await save_agent_steps(run_id, _collect_generation_steps(generation_events))
            return ChatResult(response=response, selected_agent=decision.agent, run_id=run_id)
        except Exception as exc:
            await self.event_bus.publish(
                EventRecord(
                    event_type=EventType.REQUEST_FAILED,
                    agent=decision.agent,
                    error=str(exc),
                )
            )
            await save_run(
                id=run_id,
                user_id=user_id,
                prompt=agent_input.task,
                status="error",
                agent=decision.agent.value,
                error=str(exc),
            )
            raise

    @staticmethod
    def _build_agent_input(request: ChatRequest) -> AgentRoutingInput:
        return AgentRoutingInput(
            task=ChatService._prompt_text(request.messages),
            preferred_agent=request.agent_hint,
            preference=request.preference,
            preferred_provider=request.preferred_provider,
        )

    @staticmethod
    def _prompt_text(messages: list[ChatMessage]) -> str:
        return "\n".join(
            f"{message.role.value.upper()}: {message.content}"
            for message in messages
        )
