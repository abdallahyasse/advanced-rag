from app.llm.groq_client import GroqClient
from app.agentic_rag.models import RewriteResult

SYSTEM_PROMPT = """
You are an AI Search Query Rewriter.

The previous retrieval failed to retrieve enough context.

Your job is to rewrite the search query so retrieval becomes easier.

Rules:

- Preserve the original meaning.
- Expand abbreviations.
- Add important keywords.
- Make the query retrieval-friendly.
- Do NOT answer the question.
- Return JSON only.

Schema:

{
    "rewritten_question":"..."
}
"""


class SearchQueryRewriter:

    def __init__(self):

        self.llm = GroqClient()

    def rewrite(
        self,
        question: str,
        reason: str,
    ) -> RewriteResult:

        prompt = f"""
{SYSTEM_PROMPT}

Original Question:

{question}

Why Retrieval Failed:

{reason}
"""

        return self.llm.chat(
            prompt=prompt,
            response_model=RewriteResult,
        )