from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.models.schemas import EventRecord


EventHandler = Callable[[EventRecord], Awaitable[None]]


@dataclass
class EventBus:
    subscribers: dict[str, list[EventHandler]] = field(default_factory=dict)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self.subscribers.setdefault(event_type, []).append(handler)

    async def publish(self, event: EventRecord) -> None:
        for handler in self.subscribers.get(event.event_type.value, []):
            await handler(event)


event_bus = EventBus()
