
from app.retrieval.bm25_service import BM25Service


def test_bm25_returns_relevant_documents():
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

    bm25 = BM25Service()
    bm25.build(documents)

    results = bm25.search("Machine Learning", top_k=3)

    assert results
    assert results[0]["text"] == documents[2]["text"]
    assert results[0]["page"] == 3
    assert results[0]["source"] == "test.pdf"


def test_bm25_search_before_build_raises_error():
    bm25 = BM25Service()

    try:
        bm25.search("Machine Learning")
    except RuntimeError as exc:
        assert "not been built" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError before building BM25.")
