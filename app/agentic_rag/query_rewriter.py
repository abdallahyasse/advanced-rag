from app.container.dependencies import container
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
- Expand abbreviations if needed.
- Use conversation history.
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
    ) -> RewriteResult:

        context = container.context_builder.build_context(
            question
        )

        prompt = f"""
{SYSTEM_PROMPT}

Conversation:

{context}
"""

        return self.llm.chat(
            prompt,
            RewriteResult,
        )