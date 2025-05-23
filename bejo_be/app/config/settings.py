from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    app_name: str = "BEJO Agentic RAG API"
    app_version: str = "2.0.0"
    debug: bool = False

    # Qdrant Settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Google AI API
    google_api_key: Optional[str] = Field(default=None)

    # LLM Settings
    default_model: str = "gemini-2.0-flash"
    default_temperature: float = 0.7

    # Agent Settings
    max_iterations: int = 5
    max_tools_per_agent: int = 10

    # Document Processing
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    max_file_size_mb: int = 100

    # CORS Settings
    cors_origins: list = ["*"]
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="allow",
    )


def get_settings() -> Settings:
    """Get settings instance"""
    return Settings()
