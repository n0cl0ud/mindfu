"""
MindFu RAG Service Configuration
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    llm_base_url: str = "http://llm:8000/v1"
    llm_model: str = "devstral-small-2"
    llm_timeout: int = 300

    # Qdrant Configuration
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    default_collection: str = "documents"

    # PostgreSQL Configuration
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "mindfu"
    postgres_password: str = "mindfu_secret"
    postgres_db: str = "mindfu"

    # Redis Configuration
    redis_host: str = "redis"
    redis_port: int = 6379

    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5

    # Logging
    log_conversations: bool = False

    # Model context limit (for error messages)
    max_model_len: int = 81920

    # Workarounds
    # Disable streaming when tools are present (vLLM Mistral parser bug)
    # Set to False for models with working streaming tool calls (e.g., Nemotron with qwen3_coder)
    force_no_stream_with_tools: bool = True

    @property
    def postgres_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def postgres_sync_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
