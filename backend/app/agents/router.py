from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import AgentRoutingDecision, AgentRoutingInput, AgentType


@dataclass(frozen=True)
class AgentRule:
    agent: AgentType
    keywords: tuple[str, ...]
    reason: str


class AgentRouter:
    def __init__(self) -> None:
        self._rules: tuple[AgentRule, ...] = (
            AgentRule(
                agent=AgentType.CODER,
                keywords=("code", "implement", "debug", "bug", "refactor", "function", "class"),
                reason="task indicates software implementation or debugging",
            ),
            AgentRule(
                agent=AgentType.ANALYST,
                keywords=("analyze", "analysis", "compare", "metrics", "trend", "dataset"),
                reason="task indicates analytical reasoning",
            ),
            AgentRule(
                agent=AgentType.DOCUMENT,
                keywords=("document", "pdf", "contract", "policy", "extract", "summarize file"),
                reason="task indicates document understanding or extraction",
            ),
            AgentRule(
                agent=AgentType.PLANNER,
                keywords=("plan", "roadmap", "strategy", "sequence", "milestone"),
                reason="task indicates planning or decomposition",
            ),
            AgentRule(
                agent=AgentType.RESEARCH,
                keywords=("research", "investigate", "find", "look up", "compare options"),
                reason="task indicates information gathering",
            ),
        )

    def route(self, request: AgentRoutingInput) -> AgentRoutingDecision:
        if request.preferred_agent is not None:
            return AgentRoutingDecision(
                agent=request.preferred_agent,
                confidence=1.0,
                reason="explicit preferred agent provided",
                metadata={
                    "task": request.task,
                    "preference": request.preference,
                    "preferred_provider": request.preferred_provider,
                },
            )

        normalized_task = request.task.lower()

        for rule in self._rules:
            if any(keyword in normalized_task for keyword in rule.keywords):
                return AgentRoutingDecision(
                    agent=rule.agent,
                    confidence=0.82,
                    reason=rule.reason,
                    metadata={
                        "matched_keywords": [
                            keyword for keyword in rule.keywords if keyword in normalized_task
                        ],
                        "task": request.task,
                        "preference": request.preference,
                        "preferred_provider": request.preferred_provider,
                    },
                )

        return AgentRoutingDecision(
            agent=AgentType.RESEARCH,
            confidence=0.5,
            reason="defaulted to research for general information tasks",
            metadata={
                "task": request.task,
                "preference": request.preference,
                "preferred_provider": request.preferred_provider,
            },
        )


agent_router = AgentRouter()
