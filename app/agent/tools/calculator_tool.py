
import ast
import math
import operator

from app.agent.models import ToolResult
from app.agent.tools.base import BaseTool


class CalculatorTool(BaseTool):
    """
    Safely evaluates basic mathematical expressions using AST.
    """

    name = "calculator"

    description = (
        "Perform arithmetic calculations. "
        "Supports +, -, *, /, //, %, ** and parentheses."
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

    _binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _unary_operators = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _evaluate(self, node):
        if isinstance(node, ast.Expression):
            return self._evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError("Only numeric constants are allowed.")

            return node.value

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)

            if operator_type not in self._binary_operators:
                raise ValueError("Unsupported mathematical operator.")

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            if operator_type is ast.Pow:
                if abs(right) > 100:
                    raise ValueError("Exponent is too large.")

            result = self._binary_operators[operator_type](left, right)

            if isinstance(result, int) and result.bit_length() > 4096:
                raise ValueError("Integer result is too large.")

            if isinstance(result, float) and not math.isfinite(result):
                raise ValueError("Result is outside the supported numeric range.")

            return result

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)

            if operator_type not in self._unary_operators:
                raise ValueError("Unsupported unary operator.")

            return self._unary_operators[operator_type](
                self._evaluate(node.operand)
            )

        raise ValueError("Only basic arithmetic expressions are allowed.")

    def run(self, expression: str) -> ToolResult:
        try:
            if not isinstance(expression, str) or not expression.strip():
                raise ValueError("Expression must not be empty.")

            if len(expression) > 256:
                raise ValueError("Expression is too long.")

            tree = ast.parse(expression, mode="eval")

            if sum(1 for _ in ast.walk(tree)) > 50:
                raise ValueError("Expression is too complex.")

            result = self._evaluate(tree)

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
