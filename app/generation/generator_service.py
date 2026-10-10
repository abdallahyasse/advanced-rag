
import re

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from app.config.settings import settings


class GeneratorService:
    """
    Evidence-based generation service.

    Responsibilities:
    - Load the language model once.
    - Extract explicitly documented facts when possible.
    - Prevent unsupported mobile-development claims.
    - Generate concise answers grounded in retrieved context.
    """

    _model: T5ForConditionalGeneration | None = None
    _tokenizer: T5Tokenizer | None = None

    INSUFFICIENT_INFORMATION = "I don't have enough information."

    @classmethod
    def get_model(cls):
        """Load the model and tokenizer once."""
        if cls._model is None:
            cls._tokenizer = T5Tokenizer.from_pretrained(
                settings.llm_model
            )

            cls._model = T5ForConditionalGeneration.from_pretrained(
                settings.llm_model
            )

            cls._model.eval()

        return cls._model, cls._tokenizer

    @staticmethod
    def _extract_degree(
        context: str,
        question: str,
    ) -> str | None:
        """Extract an explicitly documented academic degree."""

        question_lower = question.lower()

        degree_keywords = (
            "degree",
            "education",
            "bachelor",
            "master",
            "phd",
            "qualification",
        )

        if not any(word in question_lower for word in degree_keywords):
            return None

        patterns = [
            (
                r"\b(Bachelor\s+of\s+Science\s+in\s+"
                r"Computer\s+Science\s*&\s*Artificial\s+Intelligence)\b"
            ),
            (
                r"\b(Bachelor\s+of\s+Science"
                r"(?:\s+in\s+[^\n•]+)?)"
            ),
            (
                r"\b(Master\s+of\s+Science"
                r"(?:\s+in\s+[^\n•]+)?)"
            ),
            (
                r"\b(Ph\.?D\.?"
                r"(?:\s+in\s+[^\n•]+)?)"
            ),
        ]

        for pattern in patterns:
            match = re.search(pattern, context, flags=re.IGNORECASE)

            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def _extract_sat_annotator_techniques(
        context: str,
        question: str,
    ) -> str | None:
        """Extract documented techniques for SAT-Annotator."""

        question_lower = question.lower()
        context_lower = context.lower()

        if "sat-annotator" not in question_lower:
            return None

        if not any(
            word in question_lower
            for word in (
                "technique",
                "method",
                "approach",
                "how",
            )
        ):
            return None

        if "sat-annotator" not in context_lower:
            return None

        evidence = []

        segmentation_pattern = (
            r"\b(?:automatic\s+and\s+manual|"
            r"manual\s+and\s+automatic)\s+"
            r"image\s+segmentation\b"
        )

        if re.search(segmentation_pattern, context, re.IGNORECASE):
            evidence.append(
                "automatic and manual image segmentation"
            )

        if (
            re.search(r"\bcomputer\s+vision\b", context, re.IGNORECASE)
            and re.search(
                r"\bgeospatial\s+image\s+analysis\b",
                context,
                re.IGNORECASE,
            )
        ):
            evidence.append(
                "Computer Vision for geospatial image analysis"
            )

        if not evidence:
            return None

        return "SAT-Annotator used " + " and ".join(evidence) + "."

    @staticmethod
    def _is_mobile_development_question(
        question: str,
    ) -> bool:
        """Identify questions about mobile application development."""

        question_lower = question.lower()

        mobile_terms = re.search(
            r"\bmobile\b|\bandroid\b|\bios\b",
            question_lower,
        )

        development_terms = re.search(
            r"\b(app|application|develop|developed|"
            r"development|build|built|programming language|"
            r"coding language|technology|technologies)\b",
            question_lower,
        )

        return bool(mobile_terms and development_terms)

    @staticmethod
    def _has_mobile_development_evidence(
        context: str,
    ) -> bool:
        """
        Require an explicit connection between a mobile app
        and its development language or framework.
        """

        language_pattern = (
            r"(?:Python|Java|Kotlin|Swift|Dart|JavaScript|"
            r"TypeScript|Flutter|React\s+Native|C\+\+|C#)"
        )

        patterns = [
            (
                r"(?is)\b(?:developed|built|implemented|created)"
                r"\b.{0,100}?\b(?:mobile|Android|iOS)\b"
                r".{0,100}?\b(?:app|application)\b"
                r".{0,100}?\b(?:using|with|in)\s+"
                + language_pattern
                + r"(?!\w)"
            ),
            (
                r"(?is)\b(?:mobile|Android|iOS)\b"
                r".{0,100}?\b(?:app|application)\b"
                r".{0,100}?\b(?:developed|built|implemented|created)"
                r"\b.{0,100}?\b(?:using|with|in)\s+"
                + language_pattern
                + r"(?!\w)"
            ),
            (
                r"(?is)\b(?:mobile|Android|iOS)\b"
                r".{0,100}?\b(?:app|application)\b"
                r".{0,100}?\b(?:development|technology|language)"
                r"\s*[:=-]\s*"
                + language_pattern
                + r"(?!\w)"
            ),
        ]

        return any(
            re.search(pattern, context)
            for pattern in patterns
        )

    @classmethod
    def generate(
        cls,
        context: str,
        question: str,
        conversation: str = "",
        max_new_tokens: int = 128,
    ) -> str:
        """Generate an answer grounded in retrieved context."""

        # 1. Extract explicit academic qualifications.
        extracted_answer = cls._extract_degree(
            context=context,
            question=question,
        )

        if extracted_answer:
            return extracted_answer

        # 2. Extract documented project techniques.
        extracted_techniques = cls._extract_sat_annotator_techniques(
            context=context,
            question=question,
        )

        if extracted_techniques:
            return extracted_techniques

        # 3. Guard against unsupported mobile-development claims.
        if cls._is_mobile_development_question(question):
            if not cls._has_mobile_development_evidence(context):
                return cls.INSUFFICIENT_INFORMATION

        # 4. Do not ask the model to answer without evidence.
        context = context[:3500].strip()

        if not context:
            return cls.INSUFFICIENT_INFORMATION

        # 5. Load the model.
        model, tokenizer = cls.get_model()

        # 6. Build the evidence-based prompt.
        prompt = f"""Answer the question using only the information in the context.

Rules:
- Use only facts explicitly supported by the context.
- Answer the specific question, not a different question about the topic.
- Do not guess or infer missing personal information.
- A programming language in a skills list does not prove it was used for a specific project.
- If the context is insufficient, answer exactly:
I don't have enough information.
- Keep the answer direct and concise.

Context:
{context}

Question:
{question}

Answer:"""

        # 7. Tokenize.
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # 8. Generate.
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                repetition_penalty=1.1,
                early_stopping=True,
            )

        # 9. Decode.
        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if not answer:
            return cls.INSUFFICIENT_INFORMATION

        return answer
