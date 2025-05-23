import logging
import threading
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
        self._pending_messages: List[BaseMessage] = []
        self._lock = threading.Lock()
        self._message_counter = 0  # FIXED: Add counter for message indexing

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

            # FIXED: Update message counter based on loaded messages
            self._message_counter = len(self._messages)
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
        with self._lock:
            # Add to local cache immediately (this is what LangChain expects)
            self._messages.append(message)
            # Add to pending queue for async storage
            self._pending_messages.append(message)

        logger.info(
            f"Message queued for storage. Queue size: {len(self._pending_messages)}"
        )

    async def _store_message_async(
        self, message: BaseMessage, message_index: int = None
    ) -> None:
        """Store message in Qdrant asynchronously - FIXED: Accept message_index parameter"""
        try:
            await self._ensure_collection_exists()

            # Prepare message data
            timestamp = datetime.now().isoformat()
            point_id = str(uuid4())

            # Create a simple vector (we don't really use vector search for chat history)
            # Using a zero vector since we're using metadata filtering
            vector = [0.0] * settings.VECTOR_SIZE

            # FIXED: Use provided message_index or increment counter
            if message_index is None:
                with self._lock:
                    message_index = self._message_counter
                    self._message_counter += 1

            payload = {
                "session_id": self.session_id,
                "type": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content,
                "timestamp": timestamp,
                "message_index": message_index,
            }

            # Store in Qdrant
            point = PointStruct(id=point_id, vector=vector, payload=payload)

            self.vector_service.client.upsert(
                collection_name=self.collection_name, points=[point]
            )

            logger.debug(f"Message stored in Qdrant for session {self.session_id}")

        except Exception as e:
            logger.error(f"Error storing message in Qdrant: {str(e)}")

    async def flush_pending_messages(self) -> None:
        """Store all pending messages to Qdrant - FIXED: Proper indexing"""
        with self._lock:
            messages_to_store = self._pending_messages.copy()
            # Calculate starting index for these messages
            starting_index = self._message_counter - len(messages_to_store)
            self._pending_messages.clear()

        if not messages_to_store:
            return

        logger.info(f"Flushing {len(messages_to_store)} pending messages to storage")

        # FIXED: Store messages with proper indexing
        for i, message in enumerate(messages_to_store):
            try:
                message_index = starting_index + i
                await self._store_message_async(message, message_index)
            except Exception as e:
                logger.error(
                    f"Failed to store message at index {starting_index + i}: {str(e)}"
                )

        logger.info(f"Successfully flushed messages for session {self.session_id}")

    async def add_message_async(self, message: BaseMessage) -> None:
        """Add a message to the chat history (async interface)"""
        await self._load_messages()

        # Add to local cache
        self._messages.append(message)

        # Store in Qdrant immediately
        await self._store_message_async(message)

    def clear(self) -> None:
        """Clear all messages for this session (synchronous interface)"""
        with self._lock:
            # Clear local cache immediately
            self._messages.clear()
            self._pending_messages.clear()
            self._message_counter = 0  # FIXED: Reset counter

        logger.info(f"Messages cleared locally for session {self.session_id}")

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
            with self._lock:
                self._messages.clear()
                self._pending_messages.clear()
                self._message_counter = 0

        except Exception as e:
            logger.error(f"Error clearing chat history: {str(e)}")

    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages for this session (synchronous property)"""
        return self._messages.copy()

    async def aget_messages(self) -> List[BaseMessage]:
        """Get all messages for this session (async method)"""
        await self._load_messages()
        return self._messages.copy()
