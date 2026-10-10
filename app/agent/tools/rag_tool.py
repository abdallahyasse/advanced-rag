from app.agent.models import ToolResult
from app.agent.tools.base import BaseTool
from app.agentic_rag.workflow import AgenticRAGWorkflow


class RAGTool(BaseTool):
    """
    Agentic RAG Tool.

    Uses the complete Agentic RAG workflow
    to answer knowledge-base questions.
    """

    name = "rag"

    description = (
        "Answer questions using the indexed knowledge base "
        "through the Agentic RAG pipeline."
    )

    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Question to answer using the knowledge base.",
            }
        },
        "required": ["question"],
    }

    def __init__(self):

        self.workflow = AgenticRAGWorkflow()

    def run(
        self,
        question: str,
    ) -> ToolResult:

        try:

            result = self.workflow.run(
                question=question,
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