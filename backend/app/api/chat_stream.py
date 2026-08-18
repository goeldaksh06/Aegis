from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agents.orchestrator import run_orchestrated_plan
from app.api.chat_service import ChatService, agent_result_to_response
from app.app_container import get_chat_service
from app.database.db import save_run
from app.models.schemas import AgentRoutingInput, AgentResult, AgentType, ChatMessage, ChatRequest

router = APIRouter()

_DONE = object()
_ERROR = object()


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _prompt_text(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{message.role.value.upper()}: {message.content}" for message in messages)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Server-Sent Events view of the exact same pipeline POST /chat runs.

    This intentionally does not duplicate agent logic — it drives the same
    ChatService.agents / agent_router used by /chat, and the same execute_agent_turn()
    (app/agents/support.py) via the optional on_stage callback, so there is exactly one
    place the request lifecycle is implemented. The only thing this endpoint adds is
    surfacing checkpoints that already exist internally (moderation, memory recall, RAG
    retrieval, LLM call, cost estimate, mission-brief construction, evaluation, persistence,
    and — when request.orchestrate is set — per-sub-agent handoffs) as they happen, instead
    of only returning the final result.
    """

    async def event_generator() -> AsyncIterator[str]:
        start = time.perf_counter()
        yield _sse("received", {"message_count": len(request.messages)})

        agent_input = AgentRoutingInput(
            task=_prompt_text(request.messages),
            preferred_agent=request.agent_hint,
            preference=request.preference,
            preferred_provider=request.preferred_provider,
        )
        decision = chat_service.agent_router.route(agent_input)
        yield _sse(
            "routed",
            {
                "agent": decision.agent.value,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
        )

        agent = chat_service.agents.get(decision.agent)
        if agent is None:
            yield _sse("error", {"message": f"No agent registered for '{decision.agent}'"})
            return

        queue: asyncio.Queue[tuple[object, object]] = asyncio.Queue()

        async def on_stage(name: str, data: dict[str, object]) -> None:
            await queue.put((name, data))

        async def run_agent() -> None:
            try:
                if request.orchestrate:
                    planner = chat_service.agents.get(AgentType.PLANNER, agent)
                    result, subs = await run_orchestrated_plan(
                        planner_agent=planner,
                        agents=chat_service.agents,
                        agent_router=chat_service.agent_router,
                        request=request,
                        on_stage=on_stage,
                    )
                else:
                    result, subs = await agent.handle(request, on_stage=on_stage), []
                await queue.put((_DONE, (result, subs)))
            except Exception as exc:  # noqa: BLE001 - surfaced to the client, not swallowed
                await queue.put((_ERROR, str(exc)))

        task = asyncio.create_task(run_agent())
        agent_result: AgentResult | None = None
        sub_results: list[AgentResult] = []
        error_message: str | None = None

        while True:
            name, data = await queue.get()
            if name is _DONE:
                agent_result, sub_results = data  # type: ignore[assignment]
                break
            if name is _ERROR:
                error_message = data  # type: ignore[assignment]
                break
            yield _sse(name, data)  # type: ignore[arg-type]

        await task

        if error_message is not None or agent_result is None:
            await save_run(prompt=agent_input.task, status="error", agent=decision.agent.value, error=error_message)
            yield _sse("error", {"message": error_message or "Agent produced no result."})
            return

        response = agent_result_to_response(agent_result, sub_results)

        await save_run(
            prompt=agent_input.task,
            status="success",
            agent=decision.agent.value,
            model=agent_result.routing.model,
            provider=agent_result.routing.provider,
            risk_level=agent_result.mission_brief.risk_level.value if agent_result.mission_brief else None,
            risk_score=agent_result.mission_brief.risk_score if agent_result.mission_brief else None,
            cost_usd=agent_result.cost.cost_usd if agent_result.cost else None,
            moderation_blocked=agent_result.moderation.blocked if agent_result.moderation else None,
            conversation_id=agent_result.conversation_id,
        )

        yield _sse(
            "persisted",
            {"total_elapsed_ms": round((time.perf_counter() - start) * 1000, 1)},
        )
        yield _sse("done", response.model_dump(mode="json"))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
