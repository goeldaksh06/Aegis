from __future__ import annotations

from dataclasses import dataclass

from app.agents.support import StageCallback, execute_agent_turn
from app.llm.service import LLMService
from app.models.schemas import AgentResult, ChatRequest
from app.tools.base import BaseTool


@dataclass(frozen=True)
class PlannerAgent:
    llm_service: LLMService
    rag_tool: BaseTool | None = None

    async def handle(self, request: ChatRequest, *, on_stage: StageCallback | None = None) -> AgentResult:
        return await execute_agent_turn(
            agent_name="planner",
            llm_service=self.llm_service,
            rag_tool=self.rag_tool,
            request=request,
            system_prompt=self._system_prompt(),
            on_stage=on_stage,
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are the Planner Agent for Aegis. Given a goal, roadmap, or multi-step "
            "objective, decompose it into a concrete, ordered sequence of milestones with "
            "clear dependencies and rough sequencing (not calendar dates unless given), state "
            "which steps block which, and call out the single highest-risk step in the plan. "
            "Prefer a specific ordered plan over general strategic narrative. Do not mention "
            "provider details."
        )
