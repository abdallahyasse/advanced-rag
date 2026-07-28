from app.persistence.faiss_persistence import FAISSRepository
from app.vectorstore.faiss_store import FAISSVectorStore

chunks = [
    "Python is a programming language.",
    "Machine Learning is a branch of Artificial Intelligence.",
    "Paris is the capital of France.",
]

store = FAISSVectorStore()
store.build(chunks)

print("Saving index...")
FAISSRepository.save(
    store,
    "indexes",
)

print("Exists:", FAISSRepository.exists("indexes"))

loaded_store = FAISSRepository.load(
    "indexes",
)

print(
    loaded_store.search(
        "What is Machine Learning?",
        top_k=3,
    )
)

# لو عايز تختبر الحذف شيل التعليق
# FAISSRepository.delete("indexes")
# print("Exists after delete:", FAISSRepository.exists("indexes"))