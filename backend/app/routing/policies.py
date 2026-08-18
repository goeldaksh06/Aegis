from dataclasses import dataclass
from typing import Literal


RoutingPreference = Literal[
    "cost",
    "speed",
    "quality",
    "balanced",
]


@dataclass(frozen=True)
class RoutingPolicy:
    preference: RoutingPreference = "balanced"

    require_tools: bool = False
    require_vision: bool = False

    preferred_provider: str | None = None