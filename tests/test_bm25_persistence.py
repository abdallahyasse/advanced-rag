
from app.persistence.bm25_persistence import BM25Repository
from app.retrieval.bm25_service import BM25Service


def test_bm25_save_and_load(tmp_path):
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

    store = BM25Service()
    store.build(documents)

    BM25Repository.save(store, str(tmp_path))

    assert BM25Repository.exists(str(tmp_path))

    loaded_store = BM25Repository.load(str(tmp_path))

    assert loaded_store.documents == documents

    results = loaded_store.search("Machine Learning", top_k=3)

    assert results
    assert results[0]["text"] == documents[1]["text"]
    assert results[0]["page"] == 2
    assert results[0]["source"] == "test.pdf"


def test_bm25_repository_exists_returns_false_for_empty_directory(tmp_path):
    assert not BM25Repository.exists(str(tmp_path))
