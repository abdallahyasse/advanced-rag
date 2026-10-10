
import numpy as np

from app.persistence.faiss_persistence import FAISSRepository
from app.vectorstore.faiss_store import FAISSVectorStore


def test_faiss_save_and_load(tmp_path):
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

    store = FAISSVectorStore()
    store.build(documents)

    FAISSRepository.save(store, str(tmp_path))

    assert FAISSRepository.exists(str(tmp_path))

    loaded_store = FAISSRepository.load(str(tmp_path))

    assert loaded_store.index is not None
    assert len(loaded_store.documents) == len(documents)
    assert loaded_store.documents == documents

    results = loaded_store.search("Machine Learning", top_k=2)

    assert results
    assert all(isinstance(result["score"], float) for result in results)


def test_faiss_repository_exists_returns_false_for_empty_directory(tmp_path):
    assert not FAISSRepository.exists(str(tmp_path))
