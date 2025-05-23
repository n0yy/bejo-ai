from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging

from ...core.agent import AgentManager
from ...services.dependencies import get_agent_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    category: Optional[int] = None
    agent_id: Optional[str] = None
    model_name: Optional[str] = "gemini-2.0-flash"


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    intermediate_steps: Optional[list] = None


@router.post("/stream/{session_id}")
async def chat_stream(
    session_id: str,
    request: ChatRequest,
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """Stream chat responses"""
    try:
        agent = agent_manager.get_or_create_agent(
            agent_id=request.agent_id, model_name=request.model_name
        )

        # Add category context if provided
        message = request.message
        if request.category:
            message = f"[Please search in knowledge level {request.category}] {message}"

        async def generate_response():
            try:
                async for chunk in agent.chat(message, session_id):
                    if chunk.strip():
                        yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: Error: {str(e)}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error(f"Error in chat stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/{session_id}", response_model=ChatResponse)
async def chat_sync(
    session_id: str,
    request: ChatRequest,
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """Synchronous chat endpoint"""
    try:
        agent = agent_manager.get_or_create_agent(
            agent_id=request.agent_id, model_name=request.model_name
        )

        result = agent.chat_sync(
            message=request.message, session_id=session_id, category=request.category
        )

        return ChatResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            intermediate_steps=result.get("intermediate_steps"),
        )

    except Exception as e:
        logger.error(f"Error in sync chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    agent_id: Optional[str] = None,
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """Clear chat history for a session"""
    try:
        agent = agent_manager.get_or_create_agent(agent_id=agent_id)
        success = agent.clear_session_history(session_id)

        return {"success": success, "message": f"Session {session_id} cleared"}

    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/summary")
async def get_session_summary(
    session_id: str,
    agent_id: Optional[str] = None,
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """Get session summary"""
    try:
        agent = agent_manager.get_or_create_agent(agent_id=agent_id)
        summary = agent.get_session_summary(session_id)

        return summary

    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def get_available_tools(
    agent_id: Optional[str] = None,
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """Get available tools for an agent"""
    try:
        agent = agent_manager.get_or_create_agent(agent_id=agent_id)
        tools = agent.get_available_tools()

        return {"tools": tools}

    except Exception as e:
        logger.error(f"Error getting tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))
