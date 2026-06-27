from src.ingestion import ingest_pdf
from src.hybrid_search import HybridSearch
from src.generator import generate_answer
from src.reranker import rerank

class AdvancedRAG:
    def __init__(self):
        self.searcher = HybridSearch()
        self.chunks_data = []
        self.ready = False

    def ingest(self, source, filename: str = "") -> int:
        new_chunks = ingest_pdf(source, filename=filename)
        if not new_chunks:
            raise ValueError("No text found in PDF.")
        self.chunks_data.extend(new_chunks)
        texts = [c["text"] for c in self.chunks_data]
        self.searcher.build(texts)
        self.ready = True
        return len(new_chunks)

    def ask(self, question: str, top_k: int = 3) -> dict:
        if not self.ready:
            raise RuntimeError("No PDF ingested yet.")

        # 1. Hybrid Search - retrieve more than needed
        results = self.searcher.search(question, top_k=top_k * 3)

        # 2. Rerank
        chunks_texts = [r["text"] for r in results]
        reranked = rerank(question, chunks_texts, top_k=top_k)

        # 3. Generate
        context = "\n\n".join([r["text"] for r in reranked])
        answer = generate_answer(context, question)

        # 4. Get metadata
        sources = []
        for r in reranked:
            for i, cd in enumerate(self.chunks_data):
                if cd["text"] == r["text"]:
                    sources.append({
                        "source": cd.get("source", "unknown"),
                        "page": cd.get("page", 0),
                        "text": r["text"][:150],
                        "score": r["score"],
                    })
                    break

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    @property
    def doc_count(self):
        sources = set(c["source"] for c in self.chunks_data)
        return len(sources)