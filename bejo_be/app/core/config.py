import os
from typing import List, Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "BEJO API"
    VERSION: str = "1.1.3"
    DESCRIPTION: str = "Advanced RAG-based document querying with agentic AI"

    # LLM Settings
    GOOGLE_API_KEY: str
    LLM_MODEL: str = "gemini-2.0-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    LLM_TEMPERATURE: float = 0.3

    # Qdrant Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    VECTOR_SIZE: int = 768

    # Chat History Settings
    CHAT_HISTORY_COLLECTION: str = "chatHistory"

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_K: int = 5

    # Collections - Each category has its own collection
    CATEGORY_COLLECTIONS: Dict[int, str] = {
        1: "bejo-knowledge-level-1",
        2: "bejo-knowledge-level-2",
        3: "bejo-knowledge-level-3",
        4: "bejo-knowledge-level-4",
    }

    # Access permissions - which collections a user level can access
    ACCESS_PERMISSIONS: Dict[int, List[str]] = {
        1: ["bejo-knowledge-level-1"],
        2: ["bejo-knowledge-level-1", "bejo-knowledge-level-2"],
        3: [
            "bejo-knowledge-level-1",
            "bejo-knowledge-level-2",
            "bejo-knowledge-level-3",
        ],
        4: [
            "bejo-knowledge-level-1",
            "bejo-knowledge-level-2",
            "bejo-knowledge-level-3",
            "bejo-knowledge-level-4",
        ],
    }

    class Config:
        env_file = ".env"


settings = Settings()
