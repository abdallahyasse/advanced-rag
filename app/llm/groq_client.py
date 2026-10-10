import json
from typing import Type

from groq import Groq
from pydantic import BaseModel

from app.config.settings import settings


class GroqClient:
    """
    Shared Groq LLM Client.

    Responsible for:
    - Sending prompts to Groq
    - Requesting structured JSON output
    - Validating the response using Pydantic
    """

    def __init__(
        self,
        model: str | None = None,
    ):
        self.client = Groq(
            api_key=settings.groq_api_key,
        )

        self.model = model or settings.planner_model

    def chat(
        self,
        prompt: str,
        response_model: Type[BaseModel],
    ):
        """
        Send a prompt to Groq and validate the JSON response
        against the provided Pydantic model.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            response_format={
                "type": "json_object",
            },
        )

        content = response.choices[0].message.content

        print("\n========== RAW LLM ==========")
        print(repr(content))
        print("=============================\n")

        if not content:
            raise ValueError(
                "Groq returned an empty response."
            )

        content = content.strip()

        # Remove accidental Markdown fences
        if content.startswith("```json"):
            content = content[7:]

        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            data = json.loads(content)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON returned by Groq: {e}\n"
                f"Raw response: {content}"
            ) from e

        return response_model.model_validate(data)