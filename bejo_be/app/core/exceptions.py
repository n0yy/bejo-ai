from fastapi import HTTPException


class BejoRAGException(Exception):
    """Base exception for BEJO RAG API"""

    pass


class CollectionNotFoundError(BejoRAGException):
    """Raised when a collection is not found"""

    pass


class EmbeddingError(BejoRAGException):
    """Raised when embedding fails"""

    pass


class ChatError(BejoRAGException):
    """Raised when chat processing fails"""

    pass


def create_http_exception(status_code: int, detail: str) -> HTTPException:
    """Create standardized HTTP exception"""
    return HTTPException(status_code=status_code, detail=detail)
