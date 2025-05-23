import logging
from datetime import datetime
from typing import List
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

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history (synchronous interface)"""
        # Add to local cache immediately (this is what LangChain expects)
        self._messages.append(message)

        # Store in Qdrant asynchronously (best effort)
        import asyncio

        try:
            # Try to run in existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a task to run in background
                loop.create_task(self._store_message_async(message))
            else:
                # Run in new event loop
                asyncio.run(self._store_message_async(message))
        except Exception as e:
            logger.error(f"Error scheduling message storage: {str(e)}")

    async def _store_message_async(self, message: BaseMessage) -> None:
        """Store message in Qdrant asynchronously"""
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
                "message_index": len(self._messages)
                - 1,  # Adjust since we already added to local
            }

            # Store in Qdrant
            point = PointStruct(id=point_id, vector=vector, payload=payload)

            self.vector_service.client.upsert(
                collection_name=self.collection_name, points=[point]
            )

            logger.debug(f"Message stored in Qdrant for session {self.session_id}")

        except Exception as e:
            logger.error(f"Error storing message in Qdrant: {str(e)}")

    async def add_message_async(self, message: BaseMessage) -> None:
        """Add a message to the chat history (async interface)"""
        await self._load_messages()

        # Add to local cache
        self._messages.append(message)

        # Store in Qdrant
        await self._store_message_async(message)

    def clear(self) -> None:
        """Clear all messages for this session (synchronous interface)"""
        # Clear local cache immediately
        self._messages.clear()

        # Clear from Qdrant asynchronously
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._clear_qdrant_async())
            else:
                asyncio.run(self._clear_qdrant_async())
        except Exception as e:
            logger.error(f"Error scheduling message clearing: {str(e)}")

    async def _clear_qdrant_async(self) -> None:
        """Clear messages from Qdrant asynchronously"""
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

            logger.debug(f"Messages cleared from Qdrant for session {self.session_id}")

        except Exception as e:
            logger.error(f"Error clearing messages from Qdrant: {str(e)}")

    async def clear_async(self) -> None:
        """Clear all messages for this session (async interface)"""
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
    def messages(self) -> List[BaseMessage]:
        """Get all messages for this session (synchronous property)"""
        # Return a copy of messages - this ensures the original list isn't modified
        return self._messages.copy()

    async def aget_messages(self) -> List[BaseMessage]:
        """Get all messages for this session (async method)"""
        await self._load_messages()
        return self._messages.copy()
