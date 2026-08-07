from app.agentic_rag.models import ReflectionResult
from app.llm.groq_client import GroqClient


SYSTEM_PROMPT = """
You are an AI Reflection Agent.

Your task is to analyze the user's question BEFORE retrieval.

Determine:

1. Is the question clear?
2. Does it need rewriting?
3. Explain your reasoning.

Return ONLY valid JSON.

Schema:

{
    "clear": true,
    "needs_rewrite": false,
    "reason": "..."
}

Examples

User:
What is machine learning?

Output:

{
    "clear": true,
    "needs_rewrite": false,
    "reason": "The question is already clear."
}

----------------------------

User:
Tell me about ML

Output:

{
    "clear": false,
    "needs_rewrite": true,
    "reason": "The abbreviation ML should be expanded."
}

Return JSON only.
"""


class ReflectionAgent:

    def __init__(self):

        self.llm = GroqClient()

    def reflect(
        self,
        question: str,
    ) -> ReflectionResult:

        prompt = f"""
{SYSTEM_PROMPT}

Question:

{question}
"""

        return self.llm.chat(
            prompt=prompt,
            response_model=ReflectionResult,
        )