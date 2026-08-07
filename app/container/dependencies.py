from app.config.settings import settings

from app.ingestion import ingest_pdf
from app.index.index_manager import IndexManager
from app.retrieval.hybrid_service import HybridRetrievalService
from app.services.rag_service import RAGService

from app.memory.conversation_memory import ConversationMemory
from app.memory.memory_service import MemoryService
from app.memory.context_builder import ContextBuilder

from app.logging.logger import get_logger

logger = get_logger(__name__)


class Container:
    """
    Application Dependency Container.
    """

    def __init__(self):

        logger.info("Loading PDF...")

        documents = ingest_pdf(
            settings.pdf_path,
            filename=settings.pdf_path,
        )

        chunks = [
            document["text"]
            for document in documents
        ]

        manager = IndexManager(
            directory=settings.index_directory,
        )

        vector_store, bm25_service = manager.initialize(
            chunks,
        )

        retriever = HybridRetrievalService(
            vector_store=vector_store,
            bm25_service=bm25_service,
        )

        # Retriever
        self.retriever = retriever

        # RAG Service
        self.rag_service = RAGService(
            retriever=retriever,
        )

        # Conversation Memory
        self.memory = ConversationMemory()

        # Memory Service
        self.memory_service = MemoryService(
            memory=self.memory,
        )

        # Context Builder
        self.context_builder = ContextBuilder(
            memory_service=self.memory_service,
        )


container = Container()