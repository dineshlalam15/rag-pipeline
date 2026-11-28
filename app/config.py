"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime configuration for the local RAG pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "rag-pipeline"
    app_env: str = "development"
    log_level: str = "INFO"
    max_upload_size_mb: int = Field(default=25, ge=1)
    allowed_extensions: str = "pdf,txt,md,docx,csv,xlsx"

    # Local Storage & Logging
    local_storage_dir: str = "./data/uploads"
    sqlite_storage_dir: str = "./data/sqlite"
    enable_file_logs: bool = False
    log_file_path: str = "./logs/rag_pipeline.log"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "rag-hybrid-index"
    pinecone_namespace: str = "default"
    pinecone_top_k: int = Field(default=20, ge=1)
    rerank_top_m: int = Field(default=5, ge=1)

    # Redis (sourced from .env)
    redis_url: str = ""
    query_cache_ttl_seconds: int | None = Field(default=None, ge=1)
    embedding_cache_ttl_seconds: int | None = Field(default=None, ge=1)

    # LLM & Chunking (sourced from .env)
    openai_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    embedding_model: str = ""
    multi_query_count: int = Field(default=3, ge=1, le=6)
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)

    # Reranker & Evaluation
    reranker_provider: str = "sentence-transformers"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str | None = None
    enable_async_evaluation: bool = True
    eval_cache_min_score: float = Field(default=0.75, ge=0.0, le=1.0)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {
            ext.strip().lower().lstrip(".")
            for ext in self.allowed_extensions.split(",")
            if ext.strip()
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
