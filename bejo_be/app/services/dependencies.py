from functools import lru_cache
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..core.retrieval import RetrievalService
from ..core.agent import AgentManager
from ..services.document_loader import DocumentLoaderService
from ..config.settings import get_settings


@lru_cache()
def get_settings():
    """Get application settings"""
    from ..config.settings import Settings

    return Settings()


@lru_cache()
def get_qdrant_client():
    """Get Qdrant client instance"""
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


@lru_cache()
def get_embeddings():
    """Get embeddings instance"""
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


@lru_cache()
def get_retrieval_service():
    """Get retrieval service instance"""
    client = get_qdrant_client()
    embeddings = get_embeddings()
    return RetrievalService(client, embeddings)


@lru_cache()
def get_agent_manager():
    """Get agent manager instance"""
    retrieval_service = get_retrieval_service()
    return AgentManager(retrieval_service)


@lru_cache()
def get_document_loader():
    """Get document loader service"""
    return DocumentLoaderService()
