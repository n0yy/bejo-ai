from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    input: str = Field(..., description="User's question or message")
    category: int = Field(..., ge=1, le=4, description="User access level (1-4)")


class EmbedRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to embed")
    category: int = Field(
        ..., ge=1, le=4, description="Category level for the document"
    )


class InitChatRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user identifier")
