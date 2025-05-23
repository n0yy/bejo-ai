from fastapi import APIRouter
from app.models.responses import HealthResponse
from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", message="BEJO RAG API is running", version=settings.VERSION
    )
