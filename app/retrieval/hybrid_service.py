from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.bm25_service import BM25Service
from app.reranker.reranker_service import RerankerService


class HybridRetrievalService:
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_service: BM25Service,
    ):
        self.vector_store = vector_store
        self.bm25_service = bm25_service

    def _rrf_fusion(
        self,
        *result_sets: list[dict],
        k: int = 60,
    ) -> list[dict]:
        fused = {}

        for results in result_sets:
            for rank, result in enumerate(results, start=1):
                index = result["index"]

                if index not in fused:
                    fused[index] = {
                        **result,
                        "rrf_score": 0.0,
                    }

                fused[index]["rrf_score"] += (
                    1.0 / (k + rank)
                )

        documents = list(fused.values())

        documents.sort(
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return documents

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        candidate_k = max(
            top_k * 2,
            10,
        )

        faiss_results = self.vector_store.search(
            query=query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_service.search(
            query=query,
            top_k=candidate_k,
        )

        bm25_results = [
            result
            for result in bm25_results
            if result.get("score", 0.0) > 0.0
        ]

        if not faiss_results and not bm25_results:
            return []

        fused_results = self._rrf_fusion(
            faiss_results,
            bm25_results,
        )

        if not fused_results:
            return []

        rerank_candidates = fused_results[
            :max(top_k * 2, 10)
        ]

        reranked = RerankerService.rerank(
            query=query,
            documents=rerank_candidates,
            top_k=top_k,
        )

        return reranked