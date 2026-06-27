import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# ── Models ────────────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Sample chunks ─────────────────────────────────────────────────────────────
chunks = [
    "FastAPI is a modern web framework for building APIs with Python.",
    "FAISS is a library for efficient similarity search of dense vectors.",
    "BM25 is a ranking function used in information retrieval.",
    "Sentence transformers convert text into dense vector embeddings.",
    "Hybrid search combines dense and sparse retrieval methods.",
    "Python is a popular programming language for data science and AI.",
]

# ── BM25 (Sparse) ─────────────────────────────────────────────────────────────
tokenized = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized)

# ── FAISS (Dense) ─────────────────────────────────────────────────────────────
vectors = embedder.encode(chunks).astype("float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# ── Hybrid Search ─────────────────────────────────────────────────────────────
def hybrid_search(query, top_k=3):
    # 1. BM25 scores
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_norm   = bm25_scores / (bm25_scores.max() + 1e-9)

    # 2. FAISS scores
    q_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, len(chunks))
    faiss_scores = np.zeros(len(chunks))
    for rank, idx in enumerate(indices[0]):
        faiss_scores[idx] = 1 - (rank / len(chunks))

    # 3. Combine
    combined = 0.5 * bm25_norm + 0.5 * faiss_scores
    top_indices = combined.argsort()[::-1][:top_k]

    return [chunks[i] for i in top_indices]

# ── Test ──────────────────────────────────────────────────────────────────────
query = "how does vector search work?"
results = hybrid_search(query)

print(f"Query: {query}\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r}")