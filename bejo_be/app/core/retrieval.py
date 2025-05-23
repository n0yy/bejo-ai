from typing import Dict, List, Optional, Any
import logging
from abc import ABC, abstractmethod

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.schema import BaseRetriever

logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies"""

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Document]:
        pass


class BasicRetrievalStrategy(RetrievalStrategy):
    """Basic similarity search strategy"""

    def __init__(self, retriever: BaseRetriever, k: int = 3):
        self.retriever = retriever
        self.k = k

    def retrieve(self, query: str, **kwargs) -> List[Document]:
        return self.retriever.get_relevant_documents(query)


class HybridRetrievalStrategy(RetrievalStrategy):
    """Hybrid retrieval combining multiple approaches"""

    def __init__(
        self, retrievers: List[BaseRetriever], weights: Optional[List[float]] = None
    ):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)

    def retrieve(self, query: str, **kwargs) -> List[Document]:
        all_docs = []
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.get_relevant_documents(query)
            # Apply weight to similarity scores if available
            for doc in docs:
                if hasattr(doc, "metadata") and "score" in doc.metadata:
                    doc.metadata["score"] *= weight
            all_docs.extend(docs)

        # Deduplicate and sort by score
        return self._deduplicate_and_rank(all_docs)

    def _deduplicate_and_rank(self, docs: List[Document]) -> List[Document]:
        """
        Deduplicate documents by their content hash and sort by score if available.

        Documents are deduplicated by taking a hash of the first 100 characters of the page content.
        If there are scores available in the metadata, the documents are sorted by score in descending order.
        The top 5 documents are returned.
        """
        seen = set()
        unique_docs = []
        for doc in docs:
            doc_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)

        # Sort by score if available
        if (
            unique_docs
            and hasattr(unique_docs[0], "metadata")
            and "score" in unique_docs[0].metadata
        ):
            unique_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)

        return unique_docs[:5]  # Return top 5


class RetrievalService:
    """Centralized retrieval service managing vector stores and strategies"""

    def __init__(self, client: QdrantClient, embeddings: GoogleGenerativeAIEmbeddings):
        self.client = client
        self.embeddings = embeddings
        self.vector_stores: Dict[str, QdrantVectorStore] = {}
        # ISA-95 Based Knowledge Hierarchy
        # Level 1: Field & Control System (only has level 1 knowledge)
        # Level 2: Supervisory (has level 1,2 knowledge)
        # Level 3: Planning (has level 1,2,3 knowledge)
        # Level 4: Management (has level 1,2,3,4 knowledge)
        self.collection_levels = {
            1: ["bejo-knowledge-level-1"],  # Field & Control System
            2: ["bejo-knowledge-level-1", "bejo-knowledge-level-2"],  # Supervisory
            3: [
                "bejo-knowledge-level-1",
                "bejo-knowledge-level-2",
                "bejo-knowledge-level-3",
            ],  # Planning
            4: [
                "bejo-knowledge-level-1",
                "bejo-knowledge-level-2",
                "bejo-knowledge-level-3",
                "bejo-knowledge-level-4",
            ],  # Management
        }

        # ISA-95 Level descriptions for better context
        self.level_descriptions = {
            1: "Field & Control System - Real-time control, sensors, actuators, basic automation",
            2: "Supervisory - SCADA, HMI, batch control, recipe management",
            3: "Planning - Production scheduling, resource allocation, workflow management",
            4: "Management - Business planning, KPIs, enterprise integration, strategic decisions",
        }
        self._initialize_vector_stores()

    def _initialize_vector_stores(self):
        """Initialize all vector stores"""
        all_collections = set()
        for collections in self.collection_levels.values():
            all_collections.update(collections)

        for collection_name in all_collections:
            try:
                self._ensure_collection_exists(collection_name)
                self.vector_stores[collection_name] = QdrantVectorStore(
                    client=self.client,
                    collection_name=collection_name,
                    embedding=self.embeddings,
                )
                logger.info(f"Initialized vector store: {collection_name}")
            except Exception as e:
                logger.error(
                    f"Failed to initialize vector store {collection_name}: {e}"
                )

    def _ensure_collection_exists(self, collection_name: str) -> bool:
        """Ensure collection exists, create if it doesn't"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if collection_name not in collection_names:
                logger.info(f"Creating collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to ensure collection {collection_name} exists: {e}")
            return False

    def get_retriever_for_level(
        self, level: int, strategy: str = "basic", **kwargs
    ) -> Optional[BaseRetriever]:
        """Get retriever for specific ISA-95 level with chosen strategy"""
        if level not in self.collection_levels:
            raise ValueError(
                f"Invalid ISA-95 level: {level}. Must be 1 (Field), 2 (Supervisory), 3 (Planning), or 4 (Management)"
            )

        collections = self.collection_levels[level]
        available_stores = [
            self.vector_stores[col] for col in collections if col in self.vector_stores
        ]

        if not available_stores:
            logger.error(
                f"No vector stores available for ISA-95 level {level} ({self.level_descriptions[level]})"
            )
            return None

        if strategy == "basic":
            # For ISA-95, use the highest level collection available for that level
            # This gives priority to more specific knowledge
            target_collection = collections[
                -1
            ]  # Last collection is the highest level available
            if target_collection in self.vector_stores:
                return self.vector_stores[target_collection].as_retriever(
                    search_kwargs=kwargs.get("search_kwargs", {"k": 3})
                )

        elif strategy == "hierarchical":
            # Use hierarchical search - start from highest level and work down
            retrievers = []
            weights = []
            for i, collection in enumerate(
                reversed(collections)
            ):  # Start from highest level
                if collection in self.vector_stores:
                    retrievers.append(
                        self.vector_stores[collection].as_retriever(
                            search_kwargs={"k": 2}
                        )
                    )
                    # Give higher weight to more specific levels
                    weights.append(1.0 + (i * 0.2))

            if retrievers:
                return HybridRetrievalStrategy(retrievers, weights)

        elif strategy == "comprehensive":
            # Use all available collections with equal weight
            retrievers = [
                store.as_retriever(search_kwargs={"k": 2}) for store in available_stores
            ]
            return HybridRetrievalStrategy(retrievers)

        return None

    def retrieve_documents(
        self, query: str, level: int, strategy: str = "basic", **kwargs
    ) -> List[Document]:
        """Main retrieval method"""
        retriever = self.get_retriever_for_level(level, strategy, **kwargs)
        if not retriever:
            return []

        try:
            if isinstance(retriever, RetrievalStrategy):
                return retriever.retrieve(query, **kwargs)
            else:
                return retriever.get_relevant_documents(query)
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            return []

    def add_documents_to_level(
        self,
        documents: List[Document],
        target_level: int,
        ids: Optional[List[str]] = None,
    ):
        """
        Add documents to specific ISA-95 level collection
        Documents are added only to the target level, not to lower levels
        """
        if target_level not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid ISA-95 target level: {target_level}")

        # Add to the specific level collection only
        collection_name = f"bejo-knowledge-level-{target_level}"
        results = {}

        if collection_name in self.vector_stores:
            try:
                self.vector_stores[collection_name].add_documents(documents, ids=ids)
                results[collection_name] = {
                    "status": "success",
                    "count": len(documents),
                    "isa_level": target_level,
                    "description": self.level_descriptions[target_level],
                }
                logger.info(
                    f"Added {len(documents)} documents to ISA-95 Level {target_level}"
                )
            except Exception as e:
                results[collection_name] = {
                    "status": "error",
                    "error": str(e),
                    "isa_level": target_level,
                }
                logger.error(
                    f"Failed to add documents to ISA-95 Level {target_level}: {e}"
                )
        else:
            results[collection_name] = {
                "status": "error",
                "error": "Collection not found",
                "isa_level": target_level,
            }

        return results

    def search_similar_documents(
        self, query: str, level: int, k: int = 5, score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents with metadata"""
        documents = self.retrieve_documents(query, level, search_kwargs={"k": k})

        results = []
        for doc in documents:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": doc.metadata.get("score", 0.0),
            }

            # Filter by score threshold
            if result["similarity_score"] >= score_threshold:
                results.append(result)

        return results

    def get_available_knowledge_for_level(self, level: int) -> Dict[str, Any]:
        """Get information about what knowledge is available for an ISA-95 level"""
        if level not in self.collection_levels:
            raise ValueError(f"Invalid ISA-95 level: {level}")

        available_collections = self.collection_levels[level]
        knowledge_info = {
            "isa_level": level,
            "level_name": self.level_descriptions[level].split(" - ")[0],
            "description": self.level_descriptions[level],
            "available_knowledge_levels": [],
            "collections": {},
        }

        for collection_name in available_collections:
            knowledge_level = int(collection_name.split("-")[-1])
            knowledge_info["available_knowledge_levels"].append(knowledge_level)

            if collection_name in self.vector_stores:
                try:
                    collection_info = self.client.get_collection(collection_name)
                    knowledge_info["collections"][f"Level_{knowledge_level}"] = {
                        "collection_name": collection_name,
                        "points_count": collection_info.points_count,
                        "status": collection_info.status,
                    }
                except Exception as e:
                    knowledge_info["collections"][f"Level_{knowledge_level}"] = {
                        "collection_name": collection_name,
                        "error": str(e),
                    }

        return knowledge_info
        """Get statistics for all collections"""
        stats = {}
        for collection_name, vector_store in self.vector_stores.items():
            try:
                collection_info = self.client.get_collection(collection_name)
                stats[collection_name] = {
                    "points_count": collection_info.points_count,
                    "vectors_count": collection_info.vectors_count,
                    "status": collection_info.status,
                }
            except Exception as e:
                stats[collection_name] = {"error": str(e)}

        return stats
