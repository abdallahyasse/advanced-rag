from app.container.dependencies import container


class RAGTool:

    name = "rag"

    description = "Answer questions using the RAG pipeline."

    def run(
        self,
        question: str,
        top_k: int = 3,
    ):

        answer = container.rag_service.ask(
            question=question,
            top_k=top_k,
        )

        return {
            "success": True,
            "result": answer,
        }