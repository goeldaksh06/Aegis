from __future__ import annotations

from dataclasses import dataclass, field

from app.models.schemas import AgentType, EventRecord


@dataclass
class EvaluationCollector:
    """Subscribes to request.completed and tracks a running average quality score per agent.

    Added to prove that new subscribers can be added later without touching the chat request
    flow — this required zero changes to ChatService.chat()'s control flow, only a new
    subscriber registered in app_container.get_chat_service(), exactly like telemetry_collector.
    """

    _score_totals: dict[str, float] = field(default_factory=dict)
    _score_counts: dict[str, int] = field(default_factory=dict)

    async def handle(self, event: EventRecord) -> None:
        evaluation = event.metadata.get("evaluation")
        if not evaluation:
            return

        agent_key = event.agent.value if isinstance(event.agent, AgentType) else str(event.agent)
        overall_score = evaluation.get("overall_score")
        if overall_score is None:
            return

        self._score_totals[agent_key] = self._score_totals.get(agent_key, 0.0) + overall_score
        self._score_counts[agent_key] = self._score_counts.get(agent_key, 0) + 1

    def average_score(self, agent: str) -> float | None:
        count = self._score_counts.get(agent, 0)
        if count == 0:
            return None
        return self._score_totals[agent] / count


evaluation_collector = EvaluationCollector()
