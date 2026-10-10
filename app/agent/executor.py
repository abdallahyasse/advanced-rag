from typing import Any

from app.agent.models import ToolResult
from app.agent.tools.registry import TOOLS


class ToolExecutor:
    """
    Executes tools registered in the Tool Registry.
    """

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:

        if tool_name not in TOOLS:

            return ToolResult(
                success=False,
                tool=tool_name,
                error=f"Unknown tool: {tool_name}",
            )

        tool = TOOLS[tool_name]

        try:

            result = tool.run(**arguments)

            if isinstance(result, ToolResult):
                return result

            return ToolResult(
                success=True,
                tool=tool_name,
                result=result,
            )

        except TypeError as e:

            return ToolResult(
                success=False,
                tool=tool_name,
                error=f"Invalid tool arguments: {e}",
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=tool_name,
                error=str(e),
            )