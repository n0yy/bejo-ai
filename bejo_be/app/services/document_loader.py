import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentLoaderService:
    """Service for loading and processing documents"""

    SUPPORTED_FORMATS = [
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".md",
        ".txt",
        ".csv",
        ".xlsx",
    ]

    def __init__(self):
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "##", "###"],
        )

    def load_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        export_type: ExportType = ExportType.MARKDOWN,
    ) -> List[Document]:
        """Load and split document"""
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format: {file_ext}")

            # Load document
            loader = DoclingLoader(file_path=file_path, export_type=export_type)
            docs = loader.load()

            if not docs:
                logger.warning(f"No content loaded from {file_path}")
                return []

            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "##", "###"],
            )

            split_docs = splitter.split_documents(docs)

            # Add source metadata
            for doc in split_docs:
                doc.metadata["source"] = file_path
                doc.metadata["file_type"] = file_ext

            logger.info(f"Loaded {len(split_docs)} chunks from {file_path}")
            return split_docs

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self.SUPPORTED_FORMATS.copy()
