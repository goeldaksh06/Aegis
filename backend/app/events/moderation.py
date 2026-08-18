from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import EventRecord


@dataclass
class ModerationCollector:
    """Subscribes to request.completed and counts blocked/flagged requests.

    Another event-bus subscriber added without touching ChatService.chat() (alongside
    telemetry_collector and evaluation_collector).
    """

    blocked_count: int = 0
    pii_flag_count: int = 0

    async def handle(self, event: EventRecord) -> None:
        moderation = event.metadata.get("moderation")
        if not moderation:
            return

        if moderation.get("blocked"):
            self.blocked_count += 1
        if moderation.get("pii_flags"):
            self.pii_flag_count += 1


moderation_collector = ModerationCollector()
