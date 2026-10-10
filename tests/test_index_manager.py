
from app.index.index_manager import IndexManager


def test_index_manager_builds_and_saves_indexes(tmp_path):
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

    manager = IndexManager(directory=str(tmp_path))

    faiss_store, bm25_store = manager.initialize(documents)

    assert faiss_store.index is not None
    assert len(faiss_store.documents) == len(documents)
    assert bm25_store.documents == documents

    assert faiss_store.documents == documents

    faiss_results = faiss_store.search("Machine Learning", top_k=3)
    bm25_results = bm25_store.search("Machine Learning", top_k=3)

    assert faiss_results
    assert bm25_results
    assert bm25_results[0]["text"] == documents[1]["text"]


def test_index_manager_loads_existing_indexes(tmp_path):
    documents = [
        {
            "text": "Python is a programming language.",
            "page": 1,
            "source": "test.pdf",
        },
        {
            "text": "Machine Learning is a branch of AI.",
            "page": 2,
            "source": "test.pdf",
        },
    ]

    manager = IndexManager(directory=str(tmp_path))

    manager.initialize(documents)

    # The second call should load the persisted indexes.
    faiss_store, bm25_store = manager.initialize(documents)

    assert faiss_store.documents == documents
    assert bm25_store.documents == documents
