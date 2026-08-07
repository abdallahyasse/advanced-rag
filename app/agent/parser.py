import json

from pydantic import BaseModel, Field, ValidationError


class ToolCall(BaseModel):
    """
    Represents a tool call produced by the LLM.
    """

    thought: str

    tool: str

    arguments: dict = Field(default_factory=dict)


class ToolParser:
    """
    Parses LLM output into a validated ToolCall.
    """

    @staticmethod
    def parse(response: str) -> ToolCall:

        try:

            data = json.loads(response)

            return ToolCall.model_validate(data)

        except json.JSONDecodeError as e:

            raise ValueError(
                f"Invalid JSON: {e}"
            )

        except ValidationError as e:

            raise ValueError(
                f"Invalid Tool Schema: {e}"
            )