from __future__ import annotations

from dataclasses import dataclass

from app.agents.support import StageCallback, execute_agent_turn
from app.llm.service import LLMService
from app.models.schemas import AgentResult, ChatRequest
from app.tools.base import BaseTool


@dataclass(frozen=True)
class AnalystAgent:
    llm_service: LLMService
    rag_tool: BaseTool | None = None

    async def handle(self, request: ChatRequest, *, on_stage: StageCallback | None = None) -> AgentResult:
        return await execute_agent_turn(
            agent_name="analyst",
            llm_service=self.llm_service,
            rag_tool=self.rag_tool,
            request=request,
            system_prompt=self._system_prompt(),
            on_stage=on_stage,
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are the Analyst Agent for Aegis. Given a crisis or operational scenario, "
            "quantify impact and risk explicitly: identify the key metrics/signals at play, "
            "compare options or scenarios where relevant, and state your risk assessment "
            "before recommending action. Prefer structured, data-grounded reasoning over "
            "general narrative. Do not mention provider details."
        )
