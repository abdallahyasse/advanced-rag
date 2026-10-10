
from app.agentic_rag.models import EvaluationResult
from app.llm.groq_client import GroqClient


SYSTEM_PROMPT = """
You are a strict, evidence-based AI Retrieval Evaluator.

Your task is to determine whether the retrieved context contains
enough explicit and relevant evidence to answer the user's exact
question accurately.

Return ONLY valid JSON matching this schema:
{
    "enough_context": true,
    "reason": "..."
}

Evaluation Rules:

1. Set enough_context=true only when the retrieved context
   explicitly supports the requested answer.

2. Distinguish relevance from sufficiency. A document may be
   related to the question without actually answering it.

3. Do not use outside knowledge, assumptions, or guesses.

4. Do not infer that a person used a technology in a specific
   project merely because it appears in their general skills list.

5. For questions asking which programming language, framework,
   library, or tool was used in a particular project, require
   explicit evidence connecting that technology to that project.

6. General experience, unrelated projects, and lists of skills
   do not prove project-specific usage.

7. If the context provides only partial evidence, or does not
   establish the requested fact, set enough_context=false.
   Explain briefly what evidence is missing.

8. If the context directly and clearly answers the question,
   set enough_context=true.

9. Evaluate the exact question provided. Do not evaluate a
   different or broader question.

10. If the context is empty or contains only whitespace,
    set enough_context=false.

11. Keep the reason concise and grounded in the retrieved context.

12. Return valid JSON only. Do not include Markdown fences
    or additional text.
"""


class EvaluatorAgent:

    def __init__(self):
        self.llm = GroqClient()

    def evaluate(
        self,
        question: str,
        context: str,
    ) -> EvaluationResult:

        # Reject empty context without calling the LLM.
        if not context or not context.strip():
            return EvaluationResult(
                enough_context=False,
                reason="The retrieved context is empty.",
            )

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
