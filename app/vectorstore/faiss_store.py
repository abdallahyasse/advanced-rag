import faiss

from app.embeddings.embedding_service import EmbeddingService


class FAISSVectorStore:
    """
    Production FAISS Vector Store.
    Responsible for:
    - Building the index
    - Storing chunks
    - Searching
    """

    def __init__(self):
        self.index: faiss.Index | None = None
        self.chunks: list[str] = []

    def build(self, chunks: list[str]) -> None:
        """
        Build the FAISS index from text chunks.
        """
        self.chunks = chunks

        vectors = EmbeddingService.embed_texts(chunks)

        dimension = vectors.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(vectors)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search the FAISS index for the most similar chunks.
        """

        if self.index is None:
            raise RuntimeError("Vector index has not been built.")

        query_vector = EmbeddingService.embed_query(query)

        distances, indices = self.index.search(query_vector, top_k)

        results = []

        for rank, idx in enumerate(indices[0]):

            # Ignore invalid FAISS results
            if idx < 0 or idx >= len(self.chunks):
                continue

            results.append(
                {
                    "text": self.chunks[idx],
                    "score": float(distances[0][rank]),
                    "index": int(idx),
                }
            )

        return results