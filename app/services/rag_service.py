from app.generation.generator_service import GeneratorService
from app.retrieval.hybrid_service import HybridRetrievalService


class RAGService:
    """
    Production RAG Service.

    Pipeline:

    Question
        ↓
    Hybrid Retrieval
        ↓
    CrossEncoder Reranker
        ↓
    Generator
        ↓
    Answer
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

        context = "\n\n".join(
            doc["text"]
            for doc in documents
        )

        answer = GeneratorService.generate(
            context=context,
            question=question,
        )

        return answer