from fastapi import HTTPException
from typing import Optional, Any, Dict


class BejoException(Exception):
    """Base exception for BEJO application"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class RetrievalException(BejoException):
    """Exception for retrieval operations"""

    pass


class AgentException(BejoException):
    """Exception for agent operations"""

    pass


class DocumentLoadException(BejoException):
    """Exception for document loading operations"""

    pass


def create_http_exception(
    status_code: int, message: str, details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create HTTP exception with details"""
    detail = {"message": message}
    if details:
        detail.update(details)

    return HTTPException(status_code=status_code, detail=detail)
