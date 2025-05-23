import logging
from uuid import uuid4
from typing import AsyncGenerator, Dict, Any
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.services.vectors import VectorService

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.supported_formats = [
            ".pdf",
            ".docx",
            ".pptx",
            ".html",
            ".md",
            ".txt",
            ".csv",
            ".xlsx",
        ]

    async def embed_document(
        self, file_path: str, category: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Embed document into appropriate collection with progress streaming"""
        try:
            # Load document
            yield {"status": "loading", "message": "Loading document..."}

            loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
            docs = loader.load()

            if not docs:
                raise EmbeddingError("No content found in the file")

            # Split documents
            yield {
                "status": "splitting",
                "message": "Splitting document into chunks...",
            }

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=["\n\n", "##", "###"],
            )
            split_docs = splitter.split_documents(docs)

            if not split_docs:
                raise EmbeddingError("Document split failed; no chunks found")

            # Get target collection for this category
            collection_name = settings.CATEGORY_COLLECTIONS.get(category)
            if not collection_name:
                raise EmbeddingError(f"Invalid category: {category}")

            # Get vector store
            vector_store = await self.vector_service.get_vector_store(collection_name)

            # Embed documents
            yield {
                "status": "embedding_started",
                "total_chunks": len(split_docs),
                "collection": collection_name,
            }

            uuids = [str(uuid4()) for _ in range(len(split_docs))]

            for i, (doc, doc_id) in enumerate(zip(split_docs, uuids)):
                try:
                    await vector_store.aadd_documents([doc], ids=[doc_id])

                    progress_percent = round(((i + 1) / len(split_docs)) * 100, 2)
                    yield {
                        "status": "progress",
                        "chunk_index": i + 1,
                        "total_chunks": len(split_docs),
                        "progress_percent": progress_percent,
                        "collection": collection_name,
                    }

                except Exception as e:
                    logger.error(f"Chunk {i+1} failed to embed: {str(e)}")
                    yield {
                        "status": "chunk_error",
                        "chunk_index": i + 1,
                        "error": str(e),
                    }

            yield {
                "status": "complete",
                "total_chunks": len(split_docs),
                "collection": collection_name,
                "category": category,
            }

        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            yield {"status": "error", "message": str(e)}
