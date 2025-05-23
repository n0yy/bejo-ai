from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import logging
from uuid import uuid4

from ...services.document_loader import DocumentLoaderService
from ...services.dependencies import get_retrieval_service, get_document_loader
from ...core.retrieval import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class EmbedRequest(BaseModel):
    file_path: str
    category: int
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200


class SearchRequest(BaseModel):
    query: str
    level: int
    k: Optional[int] = 5
    score_threshold: Optional[float] = 0.7


@router.post("/embed")
async def embed_document(
    request: EmbedRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    doc_loader: DocumentLoaderService = Depends(get_document_loader),
):
    """Embed documents with progress streaming"""
    try:
        if request.category not in [1, 2, 3, 4]:
            raise HTTPException(
                status_code=400, detail="Invalid category. Must be 1, 2, 3, or 4"
            )

        # Validate file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=404, detail=f"File not found: {request.file_path}"
            )

        def progress_stream():
            try:
                # Load documents
                yield f"data: {json.dumps({'status': 'loading', 'message': 'Loading document...'})}\n\n"

                docs = doc_loader.load_document(
                    file_path=request.file_path,
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap,
                )

                if not docs:
                    yield f"data: {json.dumps({'status': 'error', 'message': 'No content found in file'})}\n\n"
                    return

                yield f"data: {json.dumps({'status': 'splitting', 'total_chunks': len(docs)})}\n\n"

                # Add documents to vector store
                uuids = [str(uuid4()) for _ in range(len(docs))]
                results = retrieval_service.add_documents_to_level(
                    documents=docs, level=request.category, ids=uuids
                )

                # Report results
                for collection_name, result in results.items():
                    if result["status"] == "success":
                        yield f"data: {json.dumps({'status': 'progress', 'collection': collection_name, 'count': result['count']})}\n\n"
                    else:
                        yield f"data: {json.dumps({'status': 'error', 'collection': collection_name, 'error': result['error']})}\n\n"

                yield f"data: {json.dumps({'status': 'complete', 'total_chunks': len(docs), 'category': request.category})}\n\n"

            except Exception as e:
                logger.error(f"Error in document embedding: {e}")
                yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(progress_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in embed endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_knowledge(
    request: SearchRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Search knowledge base"""
    try:
        results = retrieval_service.search_similar_documents(
            query=request.query,
            level=request.level,
            k=request.k,
            score_threshold=request.score_threshold,
        )

        return {
            "query": request.query,
            "level": request.level,
            "results": results,
            "count": len(results),
        }

    except Exception as e:
        logger.error(f"Error in knowledge search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_knowledge_stats(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Get knowledge base statistics"""
    try:
        stats = retrieval_service.get_collection_stats()
        return {"collections": stats}

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
