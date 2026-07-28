from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.bm25_service import BM25Service
from app.retrieval.hybrid_service import HybridRetrievalService

chunks = [
    "Python is a programming language.",
    "Paris is the capital of France.",
    "Machine Learning is a branch of AI.",
]

vector_store = FAISSVectorStore()
vector_store.build(chunks)

bm25 = BM25Service()
bm25.build(chunks)

hybrid = HybridRetrievalService(
    vector_store,
    bm25,
)

results = hybrid.search(
    "What is AI?",
    top_k=3,
)

print(results)