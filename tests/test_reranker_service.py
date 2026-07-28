from app.reranker.reranker_service import RerankerService

documents = [
    {"text": "Python is a programming language."},
    {"text": "Paris is the capital of France."},
    {"text": "Machine Learning is a branch of AI."},
]

results = RerankerService.rerank(
    query="What is AI?",
    documents=documents,
)

print(results)