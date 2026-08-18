from __future__ import annotations

from dataclasses import dataclass

from app.agents.support import StageCallback, execute_agent_turn
from app.llm.service import LLMService
from app.models.schemas import AgentResult, ChatRequest
from app.tools.base import BaseTool


@dataclass(frozen=True)
class DocumentAgent:
    llm_service: LLMService
    rag_tool: BaseTool | None = None

    async def handle(self, request: ChatRequest, *, on_stage: StageCallback | None = None) -> AgentResult:
        return await execute_agent_turn(
            agent_name="document",
            llm_service=self.llm_service,
            rag_tool=self.rag_tool,
            request=request,
            system_prompt=self._system_prompt(),
            on_stage=on_stage,
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are the Document Agent for Aegis. Given a request to extract, summarize, or "
            "interpret a document, policy, or contract, identify the key clauses, obligations, "
            "or facts precisely, quote or reference the specific source material where "
            "available (see any retrieved context below), and flag ambiguous or missing "
            "information rather than guessing at it. Do not mention provider details."
        )
