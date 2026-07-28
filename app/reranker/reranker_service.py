from sentence_transformers import CrossEncoder

from app.config.settings import settings


class RerankerService:
    """
    Production Cross Encoder Reranker.
    """

    _model: CrossEncoder | None = None

    @classmethod
    def get_model(cls):

        if cls._model is None:
            cls._model = CrossEncoder(
                settings.reranker_model
            )

        return cls._model

    @classmethod
    def rerank(
        cls,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not documents:
            return []

        model = cls.get_model()

        pairs = [
            (query, doc["text"])
            for doc in documents
        ]

        scores = model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return documents[:top_k]