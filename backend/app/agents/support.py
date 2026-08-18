from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from app.agents.mission_brief import build_mission_brief
from app.config.settings import settings
from app.conversation.history import load_prior_messages, persist_turn
from app.eval.evaluator import evaluate_response
from app.eval.judge import judge_response
from app.llm.pricing import estimate_cost_usd
from app.llm.service import LLMService
from app.moderation.moderator import check_prompt_safety, screen_response_for_pii
from app.models.schemas import (
    AgentResult,
    AgentType,
    ChatMessage,
    ChatRequest,
    CostEstimate,
    ModerationResult,
    ProviderCallRequest,
    RoutingResult,
    ToolRequest,
    ToolType,
)
from app.routing.policies import RoutingPolicy
from app.tools.base import BaseTool

# Called at each meaningful checkpoint inside execute_agent_turn with a stage name and a
# small JSON-safe payload describing what just happened. Used to power the live SSE
# execution trace (app/api/chat_stream.py) without coupling this module to HTTP/streaming
# concerns — a no-op callback (or None) makes execute_agent_turn behave exactly as before.
StageCallback = Callable[[str, dict[str, object]], Awaitable[None]]


def build_prompt(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{message.role.value.upper()}: {message.content}" for message in messages)


def _sanitize_provider_text(text: str) -> str:
    """Strip U+FFFD replacement characters some gateways leak into responses.

    Some OpenAI-compatible gateways (observed with a MisterPilot/DeepSeek route) send
    already-invalid UTF-8 for certain punctuation (e.g. em dashes); by the time it's
    decoded the original character is unrecoverable, so this just removes the visible
    "�" glyph instead of leaving it in front of the user.
    """
    return text.replace("�", "-")


async def _emit(on_stage: StageCallback | None, name: str, data: dict[str, object]) -> None:
    if on_stage is not None:
        await on_stage(name, data)


async def execute_agent_turn(
    *,
    agent_name: str,
    llm_service: LLMService,
    rag_tool: BaseTool | None,
    request: ChatRequest,
    system_prompt: str,
    on_stage: StageCallback | None = None,
) -> AgentResult:
    """Shared execution path for prompt-in/brief-out agents (e.g. research, analyst).

    Agents differ only in their system prompt and routing keywords (app/agents/router.py) —
    RAG lookup, LLM invocation, and mission-brief derivation are identical, so that logic
    lives here once instead of being copy-pasted per agent.
    """
    current_turn_prompt = build_prompt(request.messages)

    # Safety gate: runs before anything else, including RAG — a blocked prompt never reaches
    # the LLM or the retrieval index at all, not just the final response.
    moderation = check_prompt_safety(current_turn_prompt)
    await _emit(
        on_stage,
        "moderation",
        {"blocked": moderation.blocked, "reason": moderation.block_reason},
    )
    if moderation.blocked:
        return AgentResult(
            content=moderation.block_reason or "This request was blocked by the safety policy.",
            routing=RoutingResult(
                agent=AgentType(agent_name),
                model="n/a",
                provider="none",
                confidence=1.0,
                reason="blocked by safety moderation before any provider call",
            ),
            moderation=moderation,
            conversation_id=request.conversation_id,
        )

    prior_messages: list[ChatMessage] = []
    if request.conversation_id:
        prior_messages = await load_prior_messages(request.conversation_id)
        await _emit(on_stage, "memory", {"prior_message_count": len(prior_messages)})

    # RAG relevance is queried on the current turn only — including the full conversation
    # history in the retrieval query would dilute it with old context. The LLM, however, gets
    # the full (capped) history so it can actually hold a conversation.
    effective_messages = [*prior_messages, *request.messages]
    llm_prompt = build_prompt(effective_messages)

    tool_results: list[dict[str, object]] = []

    if rag_tool is not None:
        rag_start = time.perf_counter()
        rag_result = await rag_tool.run(
            ToolRequest(tool_type=ToolType.RAG, input=current_turn_prompt, metadata={"top_k": 5})
        )
        tool_results.append(rag_result.model_dump())

        if rag_result.success and rag_result.output.strip():
            system_prompt = f"{system_prompt}\n\nRelevant retrieved context:\n{rag_result.output}"

        await _emit(
            on_stage,
            "retrieval",
            {
                "retrieved_count": rag_result.metadata.get("retrieved_count", 0),
                "sources": sorted(
                    {
                        chunk.get("metadata", {}).get("source", "unknown")
                        for chunk in rag_result.metadata.get("chunks", [])
                    }
                ),
                "elapsed_ms": round((time.perf_counter() - rag_start) * 1000, 1),
            },
        )

    llm_start = time.perf_counter()
    execution = await llm_service.generate(
        ProviderCallRequest(
            prompt=llm_prompt,
            model=request.model_hint or "router-selected",
            system_prompt=system_prompt,
            provider=request.preferred_provider,
            metadata={"agent": agent_name, "message_count": len(effective_messages)},
        ),
        RoutingPolicy(
            preference=request.preference,
            preferred_provider=request.preferred_provider or settings.MODEL_PROVIDER,
        ),
        system_prompt=system_prompt,
    )
    await _emit(
        on_stage,
        "generation",
        {
            "provider": execution.routing.provider,
            "model": execution.routing.model,
            "elapsed_ms": round((time.perf_counter() - llm_start) * 1000, 1),
            "output_tokens": execution.response.output_tokens,
        },
    )

    cost = CostEstimate(
        input_tokens=execution.response.input_tokens or 0,
        output_tokens=execution.response.output_tokens or 0,
        cost_usd=estimate_cost_usd(
            execution.routing.model, execution.response.input_tokens, execution.response.output_tokens
        ),
        model=execution.routing.model,
        provider=execution.routing.provider,
    )
    await _emit(on_stage, "cost", {"cost_usd": cost.cost_usd, "model": cost.model})

    content = _sanitize_provider_text(execution.response.content)
    moderation = moderation.model_copy(update={"pii_flags": screen_response_for_pii(content)})

    mission_brief = build_mission_brief(content)
    await _emit(
        on_stage,
        "brief",
        {
            "risk_score": mission_brief.risk_score if mission_brief else None,
            "risk_level": mission_brief.risk_level.value if mission_brief else None,
        },
    )

    evaluation = evaluate_response(content, mission_brief, tool_results)

    if request.evaluate_with_llm:
        evaluation = evaluation.model_copy(
            update={
                "judge_score": await judge_response(
                    llm_service,
                    task=current_turn_prompt,
                    response_content=content,
                    provider=execution.routing.provider,
                )
            }
        )

    await _emit(
        on_stage,
        "evaluated",
        {
            "overall_score": evaluation.overall_score,
            "groundedness": evaluation.groundedness,
            "structure_quality": evaluation.structure_quality,
        },
    )

    if request.conversation_id:
        await persist_turn(request.conversation_id, request.messages[-1].content, content)

    return AgentResult(
        content=content,
        routing=execution.routing,
        telemetry=None,
        tool_results=tool_results,
        mission_brief=mission_brief,
        evaluation=evaluation,
        cost=cost,
        moderation=moderation,
        conversation_id=request.conversation_id,
    )
