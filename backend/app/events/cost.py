from __future__ import annotations

from dataclasses import dataclass, field

from app.models.schemas import EventRecord


@dataclass
class CostCollector:
    """Subscribes to request.completed and tracks cumulative estimated spend per agent.

    Same pattern as EvaluationCollector (app/events/evaluation.py) — another subscriber added
    without touching ChatService.chat()'s control flow, this time for a FinOps concern instead
    of a quality concern.
    """

    _totals: dict[str, float] = field(default_factory=dict)
    _request_counts: dict[str, int] = field(default_factory=dict)

    async def handle(self, event: EventRecord) -> None:
        cost = event.metadata.get("cost")
        if not cost:
            return

        agent_key = event.agent.value if hasattr(event.agent, "value") else str(event.agent)
        self._totals[agent_key] = self._totals.get(agent_key, 0.0) + cost.get("cost_usd", 0.0)
        self._request_counts[agent_key] = self._request_counts.get(agent_key, 0) + 1

    @property
    def total_cost_usd(self) -> float:
        return round(sum(self._totals.values()), 6)

    def breakdown(self) -> dict[str, float]:
        return dict(self._totals)


cost_collector = CostCollector()
