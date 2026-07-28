from app.persistence.bm25_persistence import BM25Repository
from app.retrieval.bm25_service import BM25Service

chunks = [
    "Python is a programming language.",
    "Machine Learning is a branch of Artificial Intelligence.",
    "Paris is the capital of France.",
]

store = BM25Service()

store.build(chunks)

print("Saving BM25...")

BM25Repository.save(
    store,
    "indexes",
)

print(
    "Exists:",
    BM25Repository.exists("indexes"),
)

loaded = BM25Repository.load(
    "indexes",
)

results = loaded.search(
    "Machine Learning",
    top_k=3,
)

print(results)

# لاختبار الحذف
# BM25Repository.delete("indexes")
# print("Exists:", BM25Repository.exists("indexes"))