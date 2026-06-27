import numpy as np
from src.bm25_search import BM25Search
from src.vector_store import VectorStore

class HybridSearch:
    def __init__(self):
        self.bm25 = BM25Search()
        self.faiss = VectorStore()
        self.chunks = []

    def build(self, chunks: list[str]):
        self.chunks = chunks
        self.bm25.build(chunks)
        self.faiss.build(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        # BM25 scores
        bm25_results = self.bm25.search(query, top_k=len(self.chunks))
        bm25_scores = np.zeros(len(self.chunks))
        for r in bm25_results:
            bm25_scores[r["index"]] = r["score"]
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)

        # FAISS scores
        faiss_results = self.faiss.search(query, top_k=len(self.chunks))
        faiss_scores = np.zeros(len(self.chunks))
        for rank, r in enumerate(faiss_results):
            faiss_scores[r["index"]] = 1 - (rank / len(self.chunks))

        # Combine
        combined = 0.5 * bm25_norm + 0.5 * faiss_scores
        top_indices = combined.argsort()[::-1][:top_k]

        return [
            {"text": self.chunks[i], "score": float(combined[i])}
            for i in top_indices
        ]