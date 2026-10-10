from app.llm.groq_client import GroqClient
from app.agentic_rag.models import RewriteResult


SYSTEM_PROMPT = """
You are an AI Query Rewriter.

You are given a conversation.

Rewrite ONLY the LAST user request so that it becomes
a complete standalone question.

Rules:

- Preserve the original meaning.
- Resolve pronouns.
- Use previous conversation when needed.
- Expand abbreviations if useful.
- Do NOT answer the question.
- Return JSON only.

Schema:

{
    "rewritten_question":"..."
}
"""


class QueryRewriter:

    def __init__(self):

        self.llm = GroqClient()

    def rewrite(
        self,
        question: str,
        conversation_context: str,
    ) -> RewriteResult:

        prompt = f"""
{SYSTEM_PROMPT}

Conversation:

{conversation_context}

Last User Question:

{question}
"""

        return self.llm.chat(
            prompt=prompt,
            response_model=RewriteResult,
        )