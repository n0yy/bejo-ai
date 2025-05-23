from pydantic import BaseModel
from typing import Dict, Any, Optional


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: Optional[list] = None


class InitChatResponse(BaseModel):
    session_id: str
    message: str


class EmbedResponse(BaseModel):
    status: str
    message: str
    chunks_processed: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
