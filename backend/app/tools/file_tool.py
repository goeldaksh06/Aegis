from __future__ import annotations

from pathlib import Path

from app.models.schemas import ToolRequest, ToolResult, ToolType
from app.tools.base import BaseTool


class FileTool(BaseTool):
    tool_type = ToolType.FILE

    async def run(self, request: ToolRequest) -> ToolResult:
        path = Path(request.input)
        if not path.exists():
            return ToolResult(
                tool_type=self.tool_type,
                output=f"File not found: {path}",
                success=False,
                metadata={"path": str(path)},
            )

        return ToolResult(
            tool_type=self.tool_type,
            output=path.read_text(encoding="utf-8"),
            metadata={"path": str(path)},
        )
