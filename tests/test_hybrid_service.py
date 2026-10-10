
from app.retrieval.bm25_service import BM25Service
from app.retrieval.hybrid_service import HybridRetrievalService
from app.vectorstore.faiss_store import FAISSVectorStore


def test_hybrid_retrieval_returns_ranked_documents():
    documents = [
        {
            "text": "Python is a programming language.",
            "page": 1,
            "source": "test.pdf",
        },
        {
            "text": "Machine Learning is a branch of Artificial Intelligence.",
            "page": 2,
            "source": "test.pdf",
        },
        {
            "text": "Paris is the capital of France.",
            "page": 3,
            "source": "test.pdf",
        },
    ]

    vector_store = FAISSVectorStore()
    vector_store.build(documents)

    bm25_service = BM25Service()
    bm25_service.build(documents)

    retriever = HybridRetrievalService(
        vector_store=vector_store,
        bm25_service=bm25_service,
    )

    results = retriever.search(
        query="What is Machine Learning?",
        top_k=3,
    )

    assert results
    assert len(results) <= 3
    assert all(isinstance(result, dict) for result in results)
    assert all("text" in result for result in results)
    assert all("index" in result for result in results)

    # The query's most relevant document should be present
    # among the returned candidates.
    result_texts = [result["text"] for result in results]
    assert documents[1]["text"] in result_texts
