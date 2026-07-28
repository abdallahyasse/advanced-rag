from rank_bm25 import BM25Okapi
import numpy as np


class BM25Service:
    """
    Production BM25 Retrieval Service.

    Responsible for:
    - Building BM25 index
    - Searching documents
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[str] = []

    def build(self, chunks: list[str]) -> None:
        """
        Build the BM25 index.
        """
        self.chunks = chunks

        tokenized_chunks = [
            chunk.lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search using BM25.
        """

        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built.")

        query_tokens = query.lower().split()

        scores = np.array(
            self.bm25.get_scores(query_tokens)
        )

        top_indices = scores.argsort()[::-1][:top_k]

        results = []

        for idx in top_indices:

            if idx < 0 or idx >= len(self.chunks):
                continue

            results.append(
                {
                    "text": self.chunks[idx],
                    "score": float(scores[idx]),
                    "index": int(idx),
                }
            )

        return results