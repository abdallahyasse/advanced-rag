from pathlib import Path

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
        Cross-Encoder Reranker
            ↓
        Context Selection
            ↓
        Generator
            ↓
        Answer + Sources
    """

    def __init__(
        self,
        retriever: HybridRetrievalService,
    ):
        self.retriever = retriever

    @staticmethod
    def _build_generation_context(
        documents: list[dict],
        max_documents: int = 5,
    ) -> str:
        """
        Build the generation context from the
        highest-ranked retrieved documents.

        The number of selected documents is configurable.
        Retrieval results remain available separately
        for debugging and source citations.
        """

        if not documents:
            return ""

        selected_documents = documents[:max_documents]

        context_parts = []

        for document in selected_documents:
            text = document.get("text", "").strip()

            if not text:
                continue

            context_parts.append(text)

        return "\n\n".join(context_parts)

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:
        """
        Execute the complete RAG pipeline.
        """

        # 1. Retrieve and rerank documents
        documents = self.retriever.search(
            query=question,
            top_k=top_k,
        )

        if not documents:
            return {
                "answer": "",
                "sources": [],
                "documents": [],
                "context": "",
            }

        # 2. Build context using up to 5 retrieved documents
        context = self._build_generation_context(
            documents=documents,
            max_documents=min(top_k, 5),
        )

        # 3. Generate an evidence-based answer
        answer = GeneratorService.generate(
            context=context,
            question=question,
        )

        # 4. Build source citations
        sources = []
        seen_sources = set()

        for document in documents:
            source = document.get("source", "Unknown")
            page = document.get("page")

            citation_key = (source, page)

            if citation_key in seen_sources:
                continue

            seen_sources.add(citation_key)

            sources.append(
                {
                    "source": Path(source).name,
                    "page": page,
                    "score": float(
                        document.get(
                            "rerank_score",
                            document.get("score", 0.0),
                        )
                    ),
                }
            )

        # 5. Return the complete RAG result
        return {
            "answer": answer,
            "sources": sources,
            "documents": documents,
            "context": context,
        }