from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4
from datetime import datetime
from app.crew.main import bejo_crew

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory storage for chat sessions (in a real application, use a database)
chat_session = {}

class Message(BaseModel):
    role: str # user or assistant
    content: str
    timestamp: datetime = None

class ChatSession(BaseModel):
    session_id: str
    messages: List[Message] = []
    created_at: datetime = None
    updated_at: datetime = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    messages: List[Message]

@router.get("", response_model=Dict[str, str])
async def create_chat_session():
    session_id = str(uuid4())
    chat_session[session_id] = ChatSession(
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    return {"session_id": session_id}

@router.post("/{session_id}", response_model=ChatResponse)
async def conversation(session_id: str, chat_request: ChatRequest):
    if session_id not in chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    session = chat_session[session_id]
    user_message = (
        Message(role="user", content=chat_request.message, timestamp=datetime.now())
    )
    session.messages.append(user_message)

    assistant_response = bejo_crew.kickoff({"query": chat_request.message})
    print(assistant_response.get("tasks_output"))
    assistant_message = (
        Message(role="assistant", content=assistant_response.raw, timestamp=datetime.now())
    )
    session.messages.append(assistant_message)
    session.updated_at = datetime.now()
    return ChatResponse(
        session_id=session_id,
        response=assistant_response,
        messages=session.messages
    )

@router.get("/{session_id}", response_model=ChatSession)
async def get_chat_session(session_id: str):
    if session_id not in chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return chat_session[session_id]

@router.delete("/{session_id}", response_model=Dict[str, str])
async def delete_chat_session(session_id: str):
    if session_id not in chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    del chat_session[session_id]
    return {"detail": f"Chat session {session_id} deleted successfully"}
