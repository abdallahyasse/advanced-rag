from app.llm.groq_client import GroqClient
from app.llm.models import ToolResponse


SYSTEM_PROMPT = """
You are an AI Tool Calling Agent.

Your job is to select exactly ONE tool for the user's request.
You MUST select one of the three registered tools below.

You NEVER answer the user's question directly.

==================================================
AVAILABLE TOOLS
==================================================

1) calculator

Use for:
- Arithmetic
- Math
- Equations
- Percentages
- Calculations

Arguments:
{
    "expression": "..."
}

--------------------------------------------------

2) rag

Use for:
- Questions about information inside the indexed knowledge base
- Questions about documents
- AI, Machine Learning, Deep Learning, NLP, and Computer Vision
- Personal or factual information that may exist in indexed documents
- Questions whose answers may or may not be present in the knowledge base

Arguments:
{
    "question": "..."
}

If the requested information might be in the knowledge base,
select rag even when you suspect the information is unavailable.
The RAG system is responsible for determining whether sufficient
evidence exists to answer.

--------------------------------------------------

3) pdf

Use for:
- PDF page count
- PDF metadata
- Inspecting a specific PDF page
- PDF document operations

Arguments:
{
    "page": 1
}

The "page" argument is optional.

==================================================
MANDATORY RULES
==================================================

- The only valid tool names are: calculator, rag, pdf.
- NEVER invent a tool name.
- NEVER select refusal, none, or any tool not listed above.
- Never answer the user's question directly.
- Return exactly one valid JSON object.
- Do not use Markdown or code fences.
- Do not include text outside the JSON object.
- Arguments must match the selected tool.
- For calculator, provide "expression".
- For rag, provide "question".
- For pdf, provide "page" only when a page number is relevant.
- If uncertain which tool to use, select rag.

==================================================
OUTPUT FORMAT
==================================================

{
    "thought": "Short reason for selecting the tool.",
    "tool": "calculator",
    "arguments": {}
}
"""


class Planner:

    def __init__(self):
        self.llm = GroqClient()

    def plan(self, user_input: str) -> ToolResponse:
        prompt = f"""
{SYSTEM_PROMPT}

User Request:
{user_input}
"""

        result = self.llm.chat(
            prompt=prompt,
            response_model=ToolResponse,
        )

        # Defensive validation: never allow an unregistered tool.
        allowed_tools = {"calculator", "rag", "pdf"}

        if result.tool not in allowed_tools:
            return ToolResponse(
                thought=(
                    "The requested tool was invalid. "
                    "Using RAG as the fallback."
                ),
                tool="rag",
                arguments={"question": user_input},
            )

        return result