import logging
from uuid import uuid4
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.models.requests import ChatRequest, InitChatRequest
from app.models.responses import InitChatResponse
from app.services.vectors import VectorService
from app.services.agent import AgentService
from app.core.exceptions import create_http_exception

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# Dependency injection
def get_vector_service() -> VectorService:
    return VectorService()


def get_agent_service(
    vector_service: VectorService = Depends(get_vector_service),
) -> AgentService:
    return AgentService(vector_service)


@router.post("/init", response_model=InitChatResponse)
async def init_chat(request: InitChatRequest):
    """Initialize a new chat session"""
    try:
        session_id = str(uuid4())
        return InitChatResponse(
            session_id=session_id, message="Chat session initialized successfully"
        )
    except Exception as e:
        logger.error(f"Error initializing chat: {str(e)}")
        raise create_http_exception(500, str(e))


@router.post("/s/{session_id}")
async def stream_chat(
    session_id: str,
    request: ChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
):
    """Stream chat response with session persistence"""
    try:

        async def generate_response():
            async for chunk in agent_service.chat_with_history(
                session_id=session_id,
                user_input=request.input,
                user_level=request.category,
            ):
                # Format for SSE
                clean_content = chunk.replace("\n", "<br>")
                yield f"data: {clean_content}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except Exception as e:
        logger.error(f"Error in stream_chat: {str(e)}")
        raise create_http_exception(500, str(e))
