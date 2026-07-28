from app.persistence.faiss_persistence import FAISSRepository
from app.persistence.bm25_persistence import BM25Repository

from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.bm25_service import BM25Service

from app.logging.logger import get_logger

logger = get_logger(__name__)


class IndexManager:
    """
    Responsible for building or loading retrieval indexes.
    """

    def __init__(
        self,
        directory: str = "indexes",
    ):
        self.directory = directory

    def initialize(
        self,
        chunks: list[str],
    ) -> tuple[FAISSVectorStore, BM25Service]:

        if (
            FAISSRepository.exists(self.directory)
            and
            BM25Repository.exists(self.directory)
        ):

            logger.info("Loading indexes...")

            faiss_store = FAISSRepository.load(
                self.directory,
            )

            bm25_store = BM25Repository.load(
                self.directory,
            )

            return (
                faiss_store,
                bm25_store,
            )

        logger.info("Building indexes...")

        faiss_store = FAISSVectorStore()

        faiss_store.build(chunks)

        bm25_store = BM25Service()

        bm25_store.build(chunks)

        FAISSRepository.save(
            faiss_store,
            self.directory,
        )

        BM25Repository.save(
            bm25_store,
            self.directory,
        )

        return (
            faiss_store,
            bm25_store,
        )