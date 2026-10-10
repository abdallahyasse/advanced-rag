
from app.vectorstore.faiss_store import FAISSVectorStore


def test_vector_store_returns_documents():
    documents = [
        {
            "text": "Python is a programming language.",
            "page": 1,
            "source": "test.pdf",
        },
        {
            "text": "Paris is the capital of France.",
            "page": 2,
            "source": "test.pdf",
        },
        {
            "text": "Machine Learning is a branch of AI.",
            "page": 3,
            "source": "test.pdf",
        },
    ]

    store = FAISSVectorStore()
    store.build(documents)

    results = store.search("Machine Learning", top_k=3)

    assert results
    assert all(isinstance(result, dict) for result in results)
    assert all("text" in result for result in results)
    assert all("index" in result for result in results)
    assert all("score" in result for result in results)


def test_vector_store_search_before_build_raises_error():
    store = FAISSVectorStore()

    try:
        store.search("Machine Learning")
    except RuntimeError as exc:
        assert "not been built" in str(exc)
    else:
        raise AssertionError(
            "Expected RuntimeError before building the vector store."
        )
