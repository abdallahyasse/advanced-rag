from typing import TypedDict


class AgentState(TypedDict):
    question: str
    rewritten_question: str
    answer: str
    needs_rewrite: bool
    enough_context: bool