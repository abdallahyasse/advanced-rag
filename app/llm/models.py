from pydantic import BaseModel, Field


class ToolResponse(BaseModel):
    """
    Structured response expected from the LLM.
    """

    thought: str = Field(
        description="Reasoning for selecting the tool."
    )

    tool: str = Field(
        description="Tool name."
    )

    arguments: dict = Field(
        default_factory=dict
    )