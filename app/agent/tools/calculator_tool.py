from app.agent.models import ToolResult
from app.agent.tools.base import BaseTool


class CalculatorTool(BaseTool):
    """
    Calculator Tool.

    Executes simple mathematical expressions.
    """

    name = "calculator"

    description = (
        "Perform arithmetic calculations. "
        "Supports +, -, *, /, %, ** and parentheses."
    )

    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to calculate.",
            }
        },
        "required": ["expression"],
    }

    def run(
        self,
        expression: str,
    ) -> ToolResult:

        try:

            result = eval(
                expression,
                {"__builtins__": {}},
                {},
            )

            return ToolResult(
                success=True,
                tool=self.name,
                result=result,
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e),
            )