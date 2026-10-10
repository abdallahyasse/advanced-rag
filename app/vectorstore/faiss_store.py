import faiss

from app.embeddings.embedding_service import EmbeddingService


class FAISSVectorStore:

    def __init__(self):

        self.index: faiss.Index | None = None

        self.documents: list[dict] = []

    def build(
        self,
        documents: list[dict],
    ) -> None:

        self.documents = documents

        texts = [
            doc["text"]
            for doc in documents
        ]

        vectors = EmbeddingService.embed_texts(
            texts
        )

        dimension = vectors.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(vectors)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        if self.index is None:
            raise RuntimeError(
                "Vector index has not been built."
            )

        query_vector = EmbeddingService.embed_query(
            query
        )

        distances, indices = self.index.search(
            query_vector,
            top_k,
        )

        results = []

        for rank, idx in enumerate(indices[0]):

            if idx < 0 or idx >= len(self.documents):
                continue

            document = self.documents[idx]

            results.append(
                {
                    "text": document["text"],
                    "page": document.get("page"),
                    "source": document.get("source"),
                    "score": float(distances[0][rank]),
                    "index": idx,
                }
            )

        return results