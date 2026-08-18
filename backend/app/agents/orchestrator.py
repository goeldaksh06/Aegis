from __future__ import annotations

from app.agents.mission_brief import NO_ACTIONS_PLACEHOLDER, build_mission_brief
from app.agents.router import AgentRouter
from app.agents.support import StageCallback
from app.eval.evaluator import evaluate_response
from app.models.schemas import (
    AgentResult,
    AgentRoutingInput,
    AgentType,
    ChatMessage,
    ChatRequest,
    CostEstimate,
    MessageRole,
)

_MAX_SUB_STEPS = 3
_ORCHESTRATION_ELIGIBLE_AGENTS = {
    AgentType.RESEARCH,
    AgentType.ANALYST,
    AgentType.CODER,
    AgentType.DOCUMENT,
}


async def run_orchestrated_plan(
    *,
    planner_agent,
    agents: dict[AgentType, object],
    agent_router: AgentRouter,
    request: ChatRequest,
    on_stage: StageCallback | None = None,
) -> tuple[AgentResult, list[AgentResult]]:
    """Real multi-agent orchestration: Planner decomposes a task, then each sub-step is
    dispatched to whichever agent actually fits it (research/analyst/coder/document),
    executed in sequence, and the results are combined into one final answer.

    This is agent-to-agent handoff, not the single-agent routing every other request in this
    project uses — the planner's job here is literally to produce work for other agents to
    execute, not just to answer the prompt itself.
    """
    plan_result = await planner_agent.handle(request, on_stage=on_stage)
    sub_results: list[AgentResult] = []

    if plan_result.mission_brief is None:
        return plan_result, sub_results

    candidate_steps = [
        step
        for step in plan_result.mission_brief.recommended_actions
        if step != NO_ACTIONS_PLACEHOLDER
    ][:_MAX_SUB_STEPS]

    combined_sections = [f"## Plan\n\n{plan_result.content}"]
    combined_tool_results: list[dict[str, object]] = list(plan_result.tool_results)

    for index, step_text in enumerate(candidate_steps, start=1):
        decision = agent_router.route(AgentRoutingInput(task=step_text))
        target_agent_type = (
            decision.agent if decision.agent in _ORCHESTRATION_ELIGIBLE_AGENTS else AgentType.RESEARCH
        )
        target_agent = agents.get(target_agent_type)

        if target_agent is None:
            continue

        if on_stage is not None:
            await on_stage(
                "substep_start",
                {"index": index, "step": step_text, "agent": target_agent_type.value},
            )

        sub_request = ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content=step_text)],
            preference=request.preference,
            preferred_provider=request.preferred_provider,
        )
        sub_result = await target_agent.handle(sub_request, on_stage=None)
        sub_results.append(sub_result)
        combined_tool_results.extend(sub_result.tool_results)

        if on_stage is not None:
            await on_stage(
                "substep_done",
                {
                    "index": index,
                    "agent": target_agent_type.value,
                    "risk_level": sub_result.mission_brief.risk_level.value
                    if sub_result.mission_brief
                    else None,
                },
            )

        combined_sections.append(
            f"## Sub-task {index}: {step_text}\n\n"
            f"*(handled by the {target_agent_type.value} agent)*\n\n{sub_result.content}"
        )

    combined_content = "\n\n".join(combined_sections)
    combined_brief = build_mission_brief(combined_content) or plan_result.mission_brief
    combined_evaluation = evaluate_response(combined_content, combined_brief, combined_tool_results)

    total_cost_usd = (plan_result.cost.cost_usd if plan_result.cost else 0.0) + sum(
        sub.cost.cost_usd for sub in sub_results if sub.cost
    )
    total_input_tokens = (plan_result.cost.input_tokens if plan_result.cost else 0) + sum(
        sub.cost.input_tokens for sub in sub_results if sub.cost
    )
    total_output_tokens = (plan_result.cost.output_tokens if plan_result.cost else 0) + sum(
        sub.cost.output_tokens for sub in sub_results if sub.cost
    )

    combined_cost = (
        CostEstimate(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=round(total_cost_usd, 6),
            model=plan_result.routing.model,
            provider=plan_result.routing.provider,
        )
        if plan_result.cost
        else None
    )

    combined_result = AgentResult(
        content=combined_content,
        routing=plan_result.routing,
        tool_results=combined_tool_results,
        mission_brief=combined_brief,
        evaluation=combined_evaluation,
        cost=combined_cost,
        moderation=plan_result.moderation,
        conversation_id=plan_result.conversation_id,
        metadata={"sub_step_count": len(sub_results)},
    )

    return combined_result, sub_results
