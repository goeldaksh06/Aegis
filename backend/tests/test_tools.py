import pytest

from app.models.schemas import ToolRequest, ToolType
from app.tools.base import ToolRegistry
from app.tools.file_tool import FileTool
from app.tools.search_tool import SearchTool
from app.tools.python_tool import PythonTool
from app.models.schemas import ToolResult


@pytest.mark.asyncio
async def test_file_tool_reports_missing_file():
    tool = FileTool()

    result = await tool.run(ToolRequest(tool_type=ToolType.FILE, input="missing.txt"))

    assert result.success is False
    assert "File not found" in result.output


@pytest.mark.asyncio
async def test_search_tool_returns_stable_placeholder_output():
    tool = SearchTool()

    result = await tool.run(ToolRequest(tool_type=ToolType.SEARCH, input="Aegis"))

    assert result.success is True
    assert "Aegis" in result.output


def test_tool_registry_resolves_registered_tools():
    registry = ToolRegistry()
    registry.register(ToolType.SEARCH, SearchTool)
    registry.register(ToolType.PYTHON, PythonTool)

    tool = registry.get(ToolType.SEARCH)

    assert isinstance(tool, SearchTool)


@pytest.mark.asyncio
async def test_python_tool_executes_and_captures_output():
    tool = PythonTool()

    code = """
def main():
    print('hello from python tool')
"""

    result = await tool.run(ToolRequest(tool_type=ToolType.PYTHON, input=code))

    assert result.success is True
    assert 'hello from python tool' in result.output


@pytest.mark.asyncio
async def test_python_tool_handles_exceptions():
    tool = PythonTool()

    code = """
def main():
    raise ValueError('boom')
"""

    result = await tool.run(ToolRequest(tool_type=ToolType.PYTHON, input=code))

    assert result.success is False
    assert 'ValueError' in result.metadata.get('error_type')
