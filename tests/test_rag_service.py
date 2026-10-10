
from app.services.rag_service import RAGService
from app.generation.generator_service import GeneratorService


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.last_query = None
        self.last_top_k = None

    def search(self, query: str, top_k: int = 5):
        self.last_query = query
        self.last_top_k = top_k
        return self.documents[:top_k]


def test_rag_service_returns_answer_context_and_sources(monkeypatch):
    documents = [
        {
            "text": "Machine Learning is a branch of Artificial Intelligence.",
            "source": "test.pdf",
            "page": 2,
            "rerank_score": 0.95,
        },
        {
            "text": "Python is a programming language.",
            "source": "test.pdf",
            "page": 1,
            "rerank_score": 0.80,
        },
        {
            "text": "Paris is the capital of France.",
            "source": "other.pdf",
            "page": 3,
            "rerank_score": 0.50,
        },
    ]

    retriever = FakeRetriever(documents)
    rag = RAGService(retriever)

    def fake_generate(context, question):
        assert "Machine Learning" in context
        assert question == "What is Machine Learning?"
        return "Machine Learning is a branch of AI."

    monkeypatch.setattr(
        GeneratorService,
        "generate",
        staticmethod(fake_generate),
    )

    result = rag.ask("What is Machine Learning?", top_k=2)

    assert isinstance(result, dict)
    assert result["answer"] == "Machine Learning is a branch of AI."
    assert len(result["documents"]) == 2
    assert "Machine Learning" in result["context"]
    assert "Python" in result["context"]

    assert len(result["sources"]) == 2

    assert result["sources"] == [
        {
            "source": "test.pdf",
            "page": 2,
            "score": 0.95,
        },
        {
            "source": "test.pdf",
            "page": 1,
            "score": 0.80,
        },
    ]

    assert retriever.last_query == "What is Machine Learning?"
    assert retriever.last_top_k == 2


def test_rag_service_handles_no_documents(monkeypatch):
    retriever = FakeRetriever([])
    rag = RAGService(retriever)

    def unexpected_generate(context, question):
        raise AssertionError("Generator should not run without documents.")

    monkeypatch.setattr(
        GeneratorService,
        "generate",
        staticmethod(unexpected_generate),
    )

    result = rag.ask("What is Machine Learning?")

    assert result == {
        "answer": "",
        "sources": [],
        "documents": [],
        "context": "",
    }
