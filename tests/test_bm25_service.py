from app.retrieval.bm25_service import BM25Service

chunks = [
    "Python is a programming language.",
    "Paris is the capital of France.",
    "Machine Learning is a branch of AI.",
]

bm25 = BM25Service()

bm25.build(chunks)

results = bm25.search("What is AI?")

print(results)