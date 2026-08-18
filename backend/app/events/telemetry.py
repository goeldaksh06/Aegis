from __future__ import annotations

from dataclasses import dataclass, field

from app.models.schemas import EventRecord


@dataclass
class TelemetryCollector:
    events: list[EventRecord] = field(default_factory=list)

    async def handle(self, event: EventRecord) -> None:
        self.events.append(event)


telemetry_collector = TelemetryCollector()
