from sentence_transformers import CrossEncoder

_model = None

def get_reranker():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model

def rerank(query: str, chunks: list[str], top_k: int = 3) -> list[dict]:
    model = get_reranker()
    pairs = [[query, chunk] for chunk in chunks]
    scores = model.predict(pairs)
    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [
        {"text": text, "score": float(score)}
        for text, score in ranked[:top_k]
    ]