
from app.reranker.reranker_service import RerankerService


class FakeCrossEncoder:
    """Deterministic model for testing reranking logic."""

    def predict(self, pairs):
        # Document 2 is most relevant, then 3, then 1.
        return [0.1, 0.9, 0.2]


def test_reranker_sorts_documents_by_relevance(monkeypatch):
    monkeypatch.setattr(
        RerankerService,
        "get_model",
        classmethod(lambda cls: FakeCrossEncoder()),
    )

    documents = [
        {"text": "An unrelated general statement."},
        {"text": "A relevant answer about artificial intelligence."},
        {"text": "Another general statement."},
    ]

    results = RerankerService.rerank(
        query="Explain the topic",
        documents=documents,
        top_k=3,
    )

    # Print diagnostic information about final ranking.
    print([
        {
            "text": doc["text"],
            "rerank_score": doc["rerank_score"],
            "rerank_rank": doc["rerank_rank"],
            "rrf_rank": doc["rrf_rank"],
            "section_boost": doc["section_boost"],
            "final_score": doc["final_score"],
        }
        for doc in results
    ])

    assert len(results) == 3

    # Verify Cross-Encoder ranking independently.
    by_rerank = sorted(
        results,
        key=lambda doc: doc["rerank_rank"],
    )

    assert by_rerank[0]["rerank_score"] == 0.9
    assert by_rerank[1]["rerank_score"] == 0.2
    assert by_rerank[2]["rerank_score"] == 0.1

    assert all("final_score" in doc for doc in results)

    # Verify final results are sorted by final_score.
    assert results == sorted(
        results,
        key=lambda doc: doc["final_score"],
        reverse=True,
    )


def test_reranker_respects_top_k(monkeypatch):
    monkeypatch.setattr(
        RerankerService,
        "get_model",
        classmethod(lambda cls: FakeCrossEncoder()),
    )

    documents = [
        {"text": "First document."},
        {"text": "Second document."},
        {"text": "Third document."},
    ]

    results = RerankerService.rerank(
        query="Explain the topic",
        documents=documents,
        top_k=2,
    )

    assert len(results) == 2
