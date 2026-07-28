from app.generation.generator_service import GeneratorService
from app.reranker.reranker_service import RerankerService
from app.retrieval.hybrid_service import HybridRetrievalService


class RAGService:
    """
    Production RAG Service.

    Pipeline:
    Query
        ↓
    Hybrid Retrieval
        ↓
    CrossEncoder Reranker
        ↓
    LLM Generator
    """

    def __init__(
        self,
        retriever: HybridRetrievalService,
    ):
        self.retriever = retriever

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> str:

        documents = self.retriever.search(
            query=question,
            top_k=top_k,
        )

        documents = RerankerService.rerank(
            query=question,
            documents=documents,
            top_k=top_k,
        )

        context = "\n\n".join(
            doc["text"]
            for doc in documents
        )

        answer = GeneratorService.generate(
            context=context,
            question=question,
        )

        return answer