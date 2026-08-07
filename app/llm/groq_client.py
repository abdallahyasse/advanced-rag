import json
from typing import Type

from groq import Groq
from pydantic import BaseModel

from app.config.settings import settings


class GroqClient:

    def __init__(self):
        self.client = Groq(
            api_key=settings.groq_api_key,
        )

    def chat(
        self,
        prompt: str,
        response_model: Type[BaseModel],
    ):

        response = self.client.chat.completions.create(
            model=settings.planner_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        content = response.choices[0].message.content

        print("\n========== RAW LLM ==========")
        print(repr(content))
        print("=============================\n")

        content = content.strip()

        if content.startswith("```json"):
            content = content[7:]

        if content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        data = json.loads(content)

        return response_model.model_validate(data)