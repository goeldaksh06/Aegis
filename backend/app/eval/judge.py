from __future__ import annotations

import re

from app.llm.service import LLMService
from app.models.schemas import ProviderCallRequest
from app.routing.policies import RoutingPolicy

_SCORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict quality judge. You will be shown a user's task and an AI assistant's "
    "response to it. Rate the response's quality and relevance to the task on a scale from "
    "0 to 10, where 0 is useless/off-topic and 10 is excellent and fully on-point. "
    "Reply with ONLY the number, nothing else."
)


async def judge_response(
    llm_service: LLMService,
    *,
    task: str,
    response_content: str,
    provider: str,
) -> float | None:
    """Optional, costed LLM-as-judge quality score.

    Deliberately separate from evaluate_response() (app/eval/evaluator.py), which is free
    and always runs — this makes a real extra LLM call and is only invoked when a request
    explicitly opts in via ChatRequest.evaluate_with_llm, and never on the mock provider
    (judging a canned response has no signal). Returns None if judging fails or is skipped,
    never raises — a broken judge call must not break the underlying chat response.
    """
    if provider == "mock":
        return None

    judge_prompt = (
        f"TASK:\n{task}\n\nRESPONSE:\n{response_content}\n\nQuality score (0-10):"
    )

    try:
        execution = await llm_service.generate(
            ProviderCallRequest(
                prompt=judge_prompt,
                model="router-selected",
                system_prompt=_JUDGE_SYSTEM_PROMPT,
                provider=provider,
                metadata={"purpose": "llm_judge"},
            ),
            RoutingPolicy(preference="balanced", preferred_provider=provider),
            system_prompt=_JUDGE_SYSTEM_PROMPT,
        )
    except Exception:
        return None

    match = _SCORE_PATTERN.search(execution.response.content)
    if not match:
        return None

    raw_score = float(match.group(1))
    return max(0.0, min(1.0, raw_score / 10))
