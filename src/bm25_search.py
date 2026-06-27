from rank_bm25 import BM25Okapi
import numpy as np

class BM25Search:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build(self, chunks: list[str]):
        self.chunks = chunks
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {"text": self.chunks[i], "score": float(scores[i]), "index": int(i)}
            for i in top_indices
        ]