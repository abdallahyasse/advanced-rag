from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """
    Standard response returned by every tool.
    """

    success: bool

    tool: str

    result: Any | None = None

    error: str | None = None


class FinalResponse(BaseModel):
    """
    Final response returned by the Tool Calling Agent.
    """

    answer: str

    tool: str

    tool_result: Any | None = None