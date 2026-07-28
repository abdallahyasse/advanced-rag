import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from app.config.settings import settings


class GeneratorService:
    """
    Production Generator Service.
    Responsible for loading the LLM and generating answers.
    """

    _model: T5ForConditionalGeneration | None = None
    _tokenizer: T5Tokenizer | None = None

    @classmethod
    def get_model(cls):

        if cls._model is None:

            cls._tokenizer = T5Tokenizer.from_pretrained(
                settings.llm_model
            )

            cls._model = T5ForConditionalGeneration.from_pretrained(
                settings.llm_model
            )

            cls._model.eval()

        return cls._model, cls._tokenizer

    @classmethod
    def generate(
        cls,
        context: str,
        question: str,
        max_new_tokens: int = 100,
    ) -> str:

        model, tokenizer = cls.get_model()

        prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context[:1000]}

Question:
{question}

Answer:
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        return tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )