import json
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from app.models.requests import EmbedRequest
from app.services.vectors import VectorService
from app.services.embedding import EmbeddingService
from app.core.exceptions import create_http_exception

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def get_vector_service() -> VectorService:
    return VectorService()


def get_embedding_service(
    vector_service: VectorService = Depends(get_vector_service),
) -> EmbeddingService:
    return EmbeddingService(vector_service)


@router.post("/embed")
async def embed_document(
    request: EmbedRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Embed document with streaming progress"""
    try:

        async def stream_embedding():
            async for progress in embedding_service.embed_document(
                request.file_path, request.category
            ):
                yield f"data: {json.dumps(progress)}\n\n"

        return StreamingResponse(stream_embedding(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error in embed_document: {str(e)}")
        raise create_http_exception(500, str(e))
