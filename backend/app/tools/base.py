from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from app.models.schemas import ToolRequest, ToolResult, ToolType


class BaseTool(ABC):
    tool_type: ToolType

    @abstractmethod
    async def run(self, request: ToolRequest) -> ToolResult:
        raise NotImplementedError


ToolFactory = Callable[[], BaseTool]


@dataclass
class ToolRegistry:
    factories: dict[ToolType, ToolFactory] = field(default_factory=dict)

    def register(self, tool_type: ToolType, factory: ToolFactory) -> None:
        self.factories[tool_type] = factory

    def get(self, tool_type: ToolType) -> BaseTool:
        factory = self.factories.get(tool_type)
        if factory is None:
            raise ValueError(f"No tool registered for '{tool_type}'")
        return factory()
