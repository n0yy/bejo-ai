import logging
from typing import Dict, List
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.exceptions import CollectionNotFoundError

logger = logging.getLogger(__name__)


class VectorService:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self._vector_stores: Dict[str, QdrantVectorStore] = {}

    async def ensure_collection_exists(self, collection_name: str) -> bool:
        """Ensure collection exists, create if it doesn't"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if collection_name not in collection_names:
                logger.info(f"Creating collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.VECTOR_SIZE, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Collection {collection_name} created successfully")
            return True
        except Exception as e:
            logger.error(
                f"Failed to ensure collection {collection_name} exists: {str(e)}"
            )
            return False

    async def get_vector_store(self, collection_name: str) -> QdrantVectorStore:
        """Get or create vector store for collection"""
        if collection_name not in self._vector_stores:
            if not await self.ensure_collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection {collection_name} could not be created"
                )

            self._vector_stores[collection_name] = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings,
            )

        return self._vector_stores[collection_name]

    async def get_accessible_vector_stores(
        self, user_level: int
    ) -> Dict[str, QdrantVectorStore]:
        """Get all vector stores accessible to user level"""
        accessible_collections = settings.ACCESS_PERMISSIONS.get(user_level, [])
        vector_stores = {}

        for collection_name in accessible_collections:
            try:
                vector_stores[collection_name] = await self.get_vector_store(
                    collection_name
                )
            except CollectionNotFoundError:
                logger.warning(f"Collection {collection_name} not accessible")

        return vector_stores
