from typing import Any

from app.agent.tools.registry import TOOLS


class ToolExecutor:
    """
    Executes tools registered in the Tool Registry.
    """

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:

        if tool_name not in TOOLS:
            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        tool = TOOLS[tool_name]

        return tool.run(**arguments)