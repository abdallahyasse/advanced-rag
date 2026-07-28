from app.vectorstore.faiss_store import FAISSVectorStore

chunks = [
    "Python is a programming language.",
    "Paris is the capital of France.",
    "Machine Learning is a branch of AI.",
]

store = FAISSVectorStore()

store.build(chunks)

results = store.search("What is AI?")

print(results)