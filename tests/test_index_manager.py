from app.index.index_manager import IndexManager

chunks = [
    "Python is a programming language.",
    "Machine Learning is a branch of Artificial Intelligence.",
    "Paris is the capital of France.",
]

manager = IndexManager()

faiss_store, bm25_store = manager.initialize(chunks)

print()

print(
    faiss_store.search(
        "Machine Learning",
        top_k=3,
    )
)

print()

print(
    bm25_store.search(
        "Machine Learning",
        top_k=3,
    )
)