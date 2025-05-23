import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue

from app.core.config import settings
from app.services.vectors import VectorService

logger = logging.getLogger(__name__)


class QdrantChatMessageHistory(BaseChatMessageHistory):
    """Chat message history implementation using Qdrant"""

    def __init__(self, session_id: str, vector_service: VectorService):
        self.session_id = session_id
        self.vector_service = vector_service
        self.collection_name = settings.CHAT_HISTORY_COLLECTION
        self._messages: List[BaseMessage] = []
        self._loaded = False

    async def _ensure_collection_exists(self):
        """Ensure chat history collection exists"""
        await self.vector_service.ensure_collection_exists(self.collection_name)

    async def _load_messages(self):
        """Load messages from Qdrant if not already loaded"""
        if self._loaded:
            return

        try:
            await self._ensure_collection_exists()

            # Search for messages with this session_id
            search_result = self.vector_service.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id", match=MatchValue(value=self.session_id)
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )

            # Sort by timestamp and reconstruct messages
            points = sorted(
                search_result[0], key=lambda x: x.payload.get("timestamp", 0)
            )

            for point in points:
                payload = point.payload
                msg_type = payload.get("type")
                content = payload.get("content", "")

                if msg_type == "human":
                    self._messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    self._messages.append(AIMessage(content=content))

            self._loaded = True
            logger.info(
                f"Loaded {len(self._messages)} messages for session {self.session_id}"
            )

        except Exception as e:
            logger.error(f"Error loading chat history: {str(e)}")
            self._messages = []
            self._loaded = True

    async def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history"""
        await self._load_messages()

        try:
            await self._ensure_collection_exists()

            # Prepare message data
            timestamp = datetime.now().isoformat()
            point_id = str(uuid4())

            # Create a simple vector (we don't really use vector search for chat history)
            # Using a zero vector since we're using metadata filtering
            vector = [0.0] * settings.VECTOR_SIZE

            payload = {
                "session_id": self.session_id,
                "type": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content,
                "timestamp": timestamp,
                "message_index": len(self._messages),
            }

            # Store in Qdrant
            point = PointStruct(id=point_id, vector=vector, payload=payload)

            self.vector_service.client.upsert(
                collection_name=self.collection_name, points=[point]
            )

            # Add to local cache
            self._messages.append(message)

        except Exception as e:
            logger.error(f"Error adding message to chat history: {str(e)}")
            # Still add to local cache even if storage fails
            self._messages.append(message)

    async def clear(self) -> None:
        """Clear all messages for this session"""
        try:
            await self._ensure_collection_exists()

            # Delete all points for this session
            self.vector_service.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="session_id", match=MatchValue(value=self.session_id)
                        )
                    ]
                ),
            )

            # Clear local cache
            self._messages.clear()

        except Exception as e:
            logger.error(f"Error clearing chat history: {str(e)}")

    @property
    async def messages(self) -> List[BaseMessage]:
        """Get all messages for this session"""
        await self._load_messages()
        return self._messages.copy()
