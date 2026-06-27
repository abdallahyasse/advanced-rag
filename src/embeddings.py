import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, show_progress_bar=False).astype("float32")

def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])