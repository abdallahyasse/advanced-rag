from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_model: str = "google/flan-t5-base"

    planner_model: str = "llama-3.3-70b-versatile"

    groq_api_key: str

    chunk_size: int = 500

    chunk_overlap: int = 50

    pdf_path: str = "data/Abdullah.yasser.pdf"

    index_directory: str = "indexes"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


settings = Settings()