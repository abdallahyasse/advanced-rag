import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.settings import settings


class EmbeddingService:
    """
    Singleton service for generating embeddings.
    """

    _model: SentenceTransformer | None = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer(settings.embedding_model)

        return cls._model

    @classmethod
    def embed_texts(cls, texts: list[str]) -> np.ndarray:
        model = cls.get_model()
        return model.encode(
            texts,
            show_progress_bar=False
        ).astype("float32")

    @classmethod
    def embed_query(cls, query: str) -> np.ndarray:
        return cls.embed_texts([query])