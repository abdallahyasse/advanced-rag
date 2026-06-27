import faiss
import numpy as np
from src.embeddings import embed_texts

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, chunks: list[str]):
        self.chunks = chunks
        vectors = embed_texts(chunks)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        from src.embeddings import embed_query
        q_vec = embed_query(query)
        distances, indices = self.index.search(q_vec, top_k)
        return [
            {"text": self.chunks[i], "score": float(distances[0][r]), "index": int(i)}
            for r, i in enumerate(indices[0])
            if i < len(self.chunks)
        ]