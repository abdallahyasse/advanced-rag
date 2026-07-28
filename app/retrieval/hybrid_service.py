from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.bm25_service import BM25Service


class HybridRetrievalService:
    """
    Hybrid Retrieval Service.

    Combines:
    - Dense Retrieval (FAISS)
    - Sparse Retrieval (BM25)
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_service: BM25Service,
    ):
        self.vector_store = vector_store
        self.bm25_service = bm25_service

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        faiss_results = self.vector_store.search(
            query=query,
            top_k=top_k,
        )

        bm25_results = self.bm25_service.search(
            query=query,
            top_k=top_k,
        )

        merged = {}

        for result in faiss_results + bm25_results:

            idx = result["index"]

            if idx not in merged:
                merged[idx] = result

        return list(merged.values())