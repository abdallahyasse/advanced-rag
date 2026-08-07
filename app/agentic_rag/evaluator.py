from app.agentic_rag.models import EvaluationResult
from app.llm.groq_client import GroqClient


SYSTEM_PROMPT = """
You are an AI Retrieval Evaluator.

Your task is to determine whether the retrieved context is sufficient to answer the user's question.

Return ONLY valid JSON.

Schema:

{
    "enough_context": true,
    "reason": "..."
}

Rules:

- If the context clearly answers the question:
    enough_context = true

- Otherwise:
    enough_context = false

Return JSON only.
"""


class EvaluatorAgent:

    def __init__(self):

        self.llm = GroqClient()

    def evaluate(
        self,
        question: str,
        context: str,
    ) -> EvaluationResult:

        prompt = f"""
{SYSTEM_PROMPT}

Question:

{question}

Retrieved Context:

{context}
"""

        return self.llm.chat(
            prompt=prompt,
            response_model=EvaluationResult,
        )