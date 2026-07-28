import pickle
from pathlib import Path

import faiss

from app.vectorstore.faiss_store import FAISSVectorStore


class FAISSRepository:
    """
    Handles saving and loading FAISS indexes.
    """

    @staticmethod
    def save(
        store: FAISSVectorStore,
        directory: str,
    ) -> None:

        path = Path(directory)

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            store.index,
            str(path / "faiss.index"),
        )

        with open(
            path / "chunks.pkl",
            "wb",
        ) as f:
            pickle.dump(store.chunks, f)

    @staticmethod
    def load(
        directory: str,
    ) -> FAISSVectorStore:

        path = Path(directory)

        store = FAISSVectorStore()

        store.index = faiss.read_index(
            str(path / "faiss.index")
        )

        with open(
            path / "chunks.pkl",
            "rb",
        ) as f:
            store.chunks = pickle.load(f)

        return store

    @staticmethod
    def exists(
        directory: str,
    ) -> bool:

        path = Path(directory)

        return (
            (path / "faiss.index").exists()
            and
            (path / "chunks.pkl").exists()
        )

    @staticmethod
    def delete(
        directory: str,
    ) -> None:

        path = Path(directory)

        index_file = path / "faiss.index"
        chunks_file = path / "chunks.pkl"

        if index_file.exists():
            index_file.unlink()

        if chunks_file.exists():
            chunks_file.unlink()