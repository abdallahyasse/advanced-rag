from src.hybrid_search import HybridSearch

chunks = [
    "FastAPI is a modern web framework for building APIs with Python.",
    "FAISS is a library for efficient similarity search of dense vectors.",
    "BM25 is a ranking function used in information retrieval.",
    "Sentence transformers convert text into dense vector embeddings.",
    "Hybrid search combines dense and sparse retrieval methods.",
    "Python is a popular programming language for data science and AI.",
]

searcher = HybridSearch()
searcher.build(chunks)

query = "how does vector search work?"
results = searcher.search(query, top_k=3)

print(f"Query: {query}\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r['text']}")
    print(f"   Score: {r['score']:.3f}\n")