from src.ingestion import ingest_pdf
from src.hybrid_search import HybridSearch

# 1. Load PDF
print("Loading PDF...")
chunks_data = ingest_pdf("data/Abdullah.yasser.pdf")
chunks = [c["text"] for c in chunks_data]
print(f"Total chunks: {len(chunks)}")

# 2. Build hybrid search
print("Building index...")
searcher = HybridSearch()
searcher.build(chunks)

# 3. Ask questions
questions = [
    "What is the person's name?",
    "What university did he graduate from?",
    "What programming languages does he know?",
]

for q in questions:
    print(f"\nQ: {q}")
    results = searcher.search(q, top_k=2)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['text'][:150]}...")