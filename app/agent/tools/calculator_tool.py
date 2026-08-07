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

    def run(self, expression: str):
        """
        Execute a mathematical expression.
        """

        try:

            result = eval(
                expression,
                {"__builtins__": {}},
                {},
            )

            return {
                "success": True,
                "result": result,
            }

        except Exception as e:

            return {
                "success": False,
                "error": str(e),
            }