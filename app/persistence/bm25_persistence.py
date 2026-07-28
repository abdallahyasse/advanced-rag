import pickle
from pathlib import Path

from app.retrieval.bm25_service import BM25Service


class BM25Repository:
    """
    Handles saving and loading BM25 indexes.
    """

    @staticmethod
    def save(
        store: BM25Service,
        directory: str,
    ) -> None:

        path = Path(directory)

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path / "bm25.pkl",
            "wb",
        ) as f:
            pickle.dump(store, f)

    @staticmethod
    def load(
        directory: str,
    ) -> BM25Service:

        path = Path(directory)

        with open(
            path / "bm25.pkl",
            "rb",
        ) as f:
            store = pickle.load(f)

        return store

    @staticmethod
    def exists(
        directory: str,
    ) -> bool:

        path = Path(directory)

        return (path / "bm25.pkl").exists()

    @staticmethod
    def delete(
        directory: str,
    ) -> None:

        path = Path(directory)

        bm25_file = path / "bm25.pkl"

        if bm25_file.exists():
            bm25_file.unlink()