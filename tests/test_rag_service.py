from app.services.rag_service import RAGService
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.bm25_service import BM25Service
from app.retrieval.hybrid_service import HybridRetrievalService

chunks = [
    "Python is a programming language.",
    "Machine Learning is a branch of Artificial Intelligence.",
    "Paris is the capital of France.",
]

vector_store = FAISSVectorStore()
vector_store.build(chunks)

bm25 = BM25Service()
bm25.build(chunks)

retriever = HybridRetrievalService(
    vector_store=vector_store,
    bm25_service=bm25,
)

rag = RAGService(retriever)

answer = rag.ask(
    "What is Machine Learning?"
)

print(answer)