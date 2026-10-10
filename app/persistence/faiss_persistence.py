import pickle
from pathlib import Path

import faiss

from app.vectorstore.faiss_store import FAISSVectorStore


class FAISSRepository:
    """
    Handles saving and loading FAISS indexes
    and document metadata.
    """

    INDEX_FILENAME = "faiss.index"
    DOCUMENTS_FILENAME = "documents.pkl"

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

        if store.index is None:
            raise RuntimeError(
                "Cannot save an empty FAISS index."
            )

        # Save FAISS index
        faiss.write_index(
            store.index,
            str(
                path / FAISSRepository.INDEX_FILENAME
            ),
        )

        # Save documents and metadata
        with open(
            path / FAISSRepository.DOCUMENTS_FILENAME,
            "wb",
        ) as f:
            pickle.dump(
                store.documents,
                f,
            )

    @staticmethod
    def load(
        directory: str,
    ) -> FAISSVectorStore:

        path = Path(directory)

        store = FAISSVectorStore()

        # Load FAISS index
        store.index = faiss.read_index(
            str(
                path / FAISSRepository.INDEX_FILENAME
            )
        )

        # Load documents and metadata
        with open(
            path / FAISSRepository.DOCUMENTS_FILENAME,
            "rb",
        ) as f:
            store.documents = pickle.load(f)

        return store

    @staticmethod
    def exists(
        directory: str,
    ) -> bool:

        path = Path(directory)

        return (
            (
                path
                / FAISSRepository.INDEX_FILENAME
            ).exists()
            and
            (
                path
                / FAISSRepository.DOCUMENTS_FILENAME
            ).exists()
        )

    @staticmethod
    def delete(
        directory: str,
    ) -> None:

        path = Path(directory)

        index_file = (
            path
            / FAISSRepository.INDEX_FILENAME
        )

        documents_file = (
            path
            / FAISSRepository.DOCUMENTS_FILENAME
        )

        if index_file.exists():
            index_file.unlink()

        if documents_file.exists():
            documents_file.unlink()