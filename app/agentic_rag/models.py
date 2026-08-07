from pydantic import BaseModel


class ReflectionResult(BaseModel):
    clear: bool
    needs_rewrite: bool
    reason: str


class RewriteResult(BaseModel):
    rewritten_question: str


class EvaluationResult(BaseModel):
    enough_context: bool
    reason: str