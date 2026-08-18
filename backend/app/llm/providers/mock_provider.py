from __future__ import annotations

import time
from dataclasses import dataclass

from app.llm.base import BaseLLMProvider, LLMResponse


_CANNED_ANALYSIS = """\
A significant supply-chain disruption is affecting the primary logistics network, creating immediate exposure across multiple regions.

Key alerts:
- Critical delay risk at the main distribution hub due to a severe capacity shortage.
- Vendor compliance exposure has increased following a regulatory review.
- Inventory shortage is reducing fulfillment capacity by an estimated 18%.

Recommended actions:
- Activate the contingency logistics plan and reroute shipments through the secondary hub.
- Escalate vendor communication to confirm revised delivery timelines within 24 hours.
- Increase monitoring cadence on the affected corridor and prepare a fallback carrier.

Supporting evidence:
- According to the latest carrier report, transit times have increased 30% week-over-week.
- The trend indicator shows sustained volatility in regional freight capacity.
- Observed delay signals align with the disruption pattern flagged last quarter.
"""


class MockProvider(BaseLLMProvider):
    def __init__(self) -> None:
        # lightweight deterministic provider for local dev and tests
        pass

    async def generate(self, prompt: str, model: str, system_prompt: str | None = None) -> LLMResponse:
        start = time.perf_counter()

        # Deterministic canned reply — no randomness, no network call, free to run in CI/dev.
        # A realistic crisis-analysis paragraph (not a literal echo) so mock mode still
        # exercises build_mission_brief() the same way a real provider's prose would.
        if "AEGIS_OK" in prompt:
            content = "AEGIS_OK"
        else:
            content = _CANNED_ANALYSIS

        latency_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            content=content,
            model=model,
            provider="mock",
            input_tokens=len(prompt.split()),
            output_tokens=len(content.split()),
            latency_ms=latency_ms,
        )
