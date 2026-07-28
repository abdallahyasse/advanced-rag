from app.embeddings.embedding_service import EmbeddingService

embedding = EmbeddingService.embed_query("Hello World")

print(embedding.shape)