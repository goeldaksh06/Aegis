from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import ToolRequest, ToolResult, ToolType
from app.tools.base import BaseTool


@dataclass(frozen=True)
class SearchTool(BaseTool):
    tool_type: ToolType = ToolType.SEARCH

    async def run(self, request: ToolRequest) -> ToolResult:
        query = request.input.strip()
        return ToolResult(
            tool_type=self.tool_type,
            output=f"Search results are not configured yet for: {query}",
            metadata={"query": query},
        )
