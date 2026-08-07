from app.llm.groq_client import GroqClient
from app.llm.models import ToolResponse


SYSTEM_PROMPT = """
You are an AI Tool Calling Agent.

Your ONLY job is to decide which tool should be executed.

You NEVER answer the user's question.

==================================================
AVAILABLE TOOLS
==================================================

1) calculator

Use when the user asks for:
- arithmetic
- math
- equations
- percentages
- calculations

Arguments:

{
    "expression":"..."
}

--------------------------------------------------

2) rag

Use when the user asks about information contained inside the indexed knowledge base.

Examples:
- AI
- Machine Learning
- Deep Learning
- NLP
- Computer Vision
- Questions about the PDF

Arguments:

{
    "question":"..."
}

--------------------------------------------------

3) pdf

Use when the user asks about:
- page number
- PDF metadata
- PDF inspection
- document operations

Arguments:

{
    "page":1
}

==================================================
OUTPUT FORMAT
==================================================

Return ONLY ONE valid JSON object.

{
    "thought":"Short reasoning explaining why this tool was selected.",
    "tool":"calculator | rag | pdf",
    "arguments":{}
}

==================================================
IMPORTANT RULES
==================================================

- Return JSON ONLY.
- Never answer the user's question.
- Never use markdown.
- Never use ```json.
- Never include any explanation outside JSON.
"""


class Planner:

    def __init__(self):

        self.llm = GroqClient()

    def plan(
        self,
        user_input: str,
    ) -> ToolResponse:

        prompt = f"""
{SYSTEM_PROMPT}

User Request:

{user_input}
"""

        return self.llm.chat(
            prompt=prompt,
            response_model=ToolResponse,
        )