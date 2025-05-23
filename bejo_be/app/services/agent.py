import logging
from typing import AsyncGenerator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.vectors import VectorService
from app.services.chat_history import QdrantChatMessageHistory

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    """Input schema for search tool"""

    query: str = Field(description="The search query")


class AgentService:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            streaming=True,
        )

    def _create_search_tool(self, user_level: int) -> StructuredTool:
        """Create async search tool based on user access level"""

        async def search_knowledge(query: str) -> str:
            """Search knowledge base based on user access level"""
            try:
                vector_stores = await self.vector_service.get_accessible_vector_stores(
                    user_level
                )

                if not vector_stores:
                    return "No accessible knowledge base found for your level."

                # Search across all accessible collections
                all_results = []
                for collection_name, vector_store in vector_stores.items():
                    retriever = vector_store.as_retriever(
                        search_kwargs={"k": settings.RETRIEVAL_K}
                    )
                    results = await retriever.ainvoke(query)
                    for result in results:
                        result.metadata["source_collection"] = collection_name
                    all_results.extend(results)

                # Sort by relevance score if available
                all_results.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)

                # Take top results
                top_results = all_results[: settings.RETRIEVAL_K * 2]

                # Format results
                context_parts = []
                for doc in top_results:
                    source_info = (
                        f"[{doc.metadata.get('source_collection', 'unknown')}]"
                    )
                    context_parts.append(f"{source_info} {doc.page_content}")

                return "\n\n".join(context_parts)

            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                return f"Search failed: {str(e)}"

        return StructuredTool.from_function(
            func=search_knowledge,
            name="search_knowledge",
            description="Search the knowledge base for relevant information based on user query",
            args_schema=SearchInput,
            coroutine=search_knowledge,  # It'll tell langchain this is async
        )

    def _get_session_history(self, session_id: str) -> QdrantChatMessageHistory:
        """Get Qdrant chat message history"""
        return QdrantChatMessageHistory(
            session_id=session_id, vector_service=self.vector_service
        )

    async def create_agent_executor(self, user_level: int) -> AgentExecutor:
        """Create agent executor with tools based on user level"""
        tools = [self._create_search_tool(user_level)]

        # React agent prompt
        prompt = PromptTemplate.from_template(
            """
You are BEJO, an intelligent assistant with access to a knowledge base.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
        """
        )

        agent = create_react_agent(self.llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
        )

    async def chat_with_history(
        self, session_id: str, user_input: str, user_level: int
    ) -> AsyncGenerator[str, None]:
        """Chat with persistent message history"""
        try:
            # Create agent executor
            agent_executor = await self.create_agent_executor(user_level)

            # Create runnable with message history
            agent_with_history = RunnableWithMessageHistory(
                agent_executor,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            # Configure session
            config = {"configurable": {"session_id": session_id}}

            # Stream response
            async for chunk in agent_with_history.astream(
                {"input": user_input}, config=config
            ):
                if "output" in chunk:
                    content = chunk["output"]
                    if content.strip():
                        yield content

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            yield f"I apologize, but I encountered an error: {str(e)}"
