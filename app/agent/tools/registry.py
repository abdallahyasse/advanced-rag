from app.agent.tools.calculator_tool import CalculatorTool
from app.agent.tools.rag_tool import RAGTool


TOOLS = {
    "calculator": CalculatorTool(),
    "rag": RAGTool(),
}