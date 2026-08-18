from __future__ import annotations

import asyncio
import contextlib
import io
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

from app.models.schemas import ToolRequest, ToolResult, ToolType
from app.tools.base import BaseTool


@dataclass
class PythonTool(BaseTool):
    tool_type: ToolType = ToolType.PYTHON

    async def run(self, request: ToolRequest) -> ToolResult:
        code = textwrap.dedent(request.input)

        # Restrict builtins to a minimal safe set
        safe_builtins = {
            'range': range,
            'len': len,
            'min': min,
            'max': max,
            'sum': sum,
            'enumerate': enumerate,
            'list': list,
            'dict': dict,
            'set': set,
            'abs': abs,
            'float': float,
            'int': int,
            'str': str,
            'print': print,
            # common exceptions so user code can raise them
            'Exception': Exception,
            'BaseException': BaseException,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'NameError': NameError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'RuntimeError': RuntimeError,
            'AttributeError': AttributeError,
            'ImportError': ImportError,
        }

        globals_: dict[str, Any] = {'__builtins__': safe_builtins}
        locals_: dict[str, Any] = {}

        stdout = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout):
                # Execute code in an asyncio-friendly way
                exec(compile(code, '<python-tool>', 'exec'), globals_, locals_)

                # If a 'main' is defined, call it (await if coroutine)
                maybe_main = locals_.get('main') or globals_.get('main')
                if callable(maybe_main):
                    if asyncio.iscoroutinefunction(maybe_main):
                        result = await maybe_main()
                        if result is not None:
                            print(result)
                    else:
                        result = maybe_main()
                        if result is not None:
                            print(result)

            output = stdout.getvalue()
            return ToolResult(tool_type=self.tool_type, output=output, success=True)

        except Exception as exc:  # pragma: no cover - tested via unit tests
            return ToolResult(
                tool_type=self.tool_type,
                output=str(exc),
                success=False,
                metadata={'error_type': type(exc).__name__},
            )
