from rank_bm25 import BM25Okapi
import numpy as np


class BM25Service:
    """
    Production BM25 Retrieval Service.

    Responsible for:
    - Building BM25 index
    - Searching documents
    - Preserving document metadata
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.documents: list[dict] = []

    def build(
        self,
        documents: list[dict],
    ) -> None:
        """
        Build the BM25 index.
        """

        self.documents = documents

        tokenized_documents = [
            document["text"].lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search using BM25.
        """

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built."
            )

        query_tokens = query.lower().split()

        scores = np.array(
            self.bm25.get_scores(
                query_tokens
            )
        )

        # Keep only documents with an actual BM25 match
        valid_indices = np.where(
            scores > 0.0
        )[0]

        if len(valid_indices) == 0:
            return []

        # Sort valid documents by BM25 score
        sorted_indices = valid_indices[
            np.argsort(
                scores[valid_indices]
            )[::-1]
        ]

        top_indices = sorted_indices[:top_k]

        results = []

        for idx in top_indices:
            idx = int(idx)

            if (
                idx < 0
                or idx >= len(self.documents)
            ):
                continue

            document = self.documents[idx]

            results.append(
                {
                    "text": document["text"],
                    "page": document.get("page"),
                    "source": document.get("source"),
                    "score": float(scores[idx]),
                    "index": idx,
                }
            )

        return results