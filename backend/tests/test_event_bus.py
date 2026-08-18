import pytest

from app.events.bus import EventBus
from app.models.schemas import EventRecord, EventType


@pytest.mark.asyncio
async def test_event_bus_invokes_subscribers_for_matching_event_type():
    bus = EventBus()
    received = []

    async def handler(event: EventRecord) -> None:
        received.append(event)

    bus.subscribe(EventType.REQUEST_COMPLETED.value, handler)

    await bus.publish(
        EventRecord(
            event_type=EventType.REQUEST_COMPLETED,
            metadata={"status": "ok"},
        )
    )

    assert len(received) == 1
    assert received[0].metadata["status"] == "ok"
