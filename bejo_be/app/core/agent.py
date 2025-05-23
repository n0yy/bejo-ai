from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from .retrieval import RetrievalService
from .tools import create_retrieval_tools
from .isa95_tools import create_isa95_specialized_tools

logger = logging.getLogger(__name__)


class BejoAgent:
    """Main agent orchestrator for BEJO RAG system"""

    def __init__(
        self, retrieval_service: RetrievalService, model_name: str = "gemini-2.0-flash"
    ):
        self.retrieval_service = retrieval_service
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=0.3, streaming=True
        )
        self.tools = create_retrieval_tools(
            retrieval_service
        ) + create_isa95_specialized_tools(retrieval_service)
        self.agent_executor = self._create_agent()
        self.chat_histories: Dict[str, ChatMessageHistory] = {}

    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent with tools"""

        # Enhanced prompt for better reasoning
        # Enhanced ISA-95 aware prompt for better reasoning
        prompt = PromptTemplate.from_template(
            """
You are BEJO, an intelligent assistant specialized in ISA-95 based manufacturing systems and document retrieval.

You have access to the following tools:
{tools}

ISA-95 KNOWLEDGE HIERARCHY:
Your knowledge base follows the ISA-95 automation pyramid structure:

Level 4 - MANAGEMENT (Business Planning & Logistics):
- Has access to ALL knowledge levels (1,2,3,4)
- Focus: Strategic planning, KPIs, enterprise integration, business decisions
- Typical topics: ROI, production targets, resource planning, compliance

Level 3 - PLANNING (Manufacturing Operations Management):
- Has access to knowledge levels (1,2,3)
- Focus: Production scheduling, resource allocation, workflow management
- Typical topics: MES, batch scheduling, material tracking, quality planning

Level 2 - SUPERVISORY (Supervisory Control):
- Has access to knowledge levels (1,2)
- Focus: SCADA, HMI, batch control, recipe management
- Typical topics: Process monitoring, alarm management, operator interfaces

Level 1 - FIELD & CONTROL (Basic Control):
- Has access to level 1 knowledge only
- Focus: Real-time control, sensors, actuators, basic automation
- Typical topics: PLC programming, instrumentation, field devices, control loops

REASONING GUIDELINES:
1. Always consider the ISA-95 context when choosing which level to search
2. For technical implementation questions → Start with Level 1-2
3. For operational questions → Focus on Level 2-3  
4. For business/strategic questions → Use Level 3-4
5. For integration questions → Use ISA-95 specialized tools
6. Use cross-level analysis when the question spans multiple levels

TOOL SELECTION STRATEGY:
- document_retrieval: For finding specific documents and information
- knowledge_search: For targeted search with similarity scoring
- isa95_level_analysis: For understanding cross-level impacts and relationships
- integration_points_analysis: For system integration and data flow questions
- isa95_compliance_check: For validating against ISA-95 standards

Always explain your reasoning about which ISA-95 level(s) you're searching and why.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""
        )

        # Create the ReAct agent
        agent = create_react_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Get or create chat history for a session"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
        return self.chat_histories[session_id]

    async def chat(self, message: str, session_id: str) -> AsyncGenerator[str, None]:
        """Chat with the agent using streaming response"""
        try:
            # Get chat history
            chat_history = self.get_session_history(session_id)

            # Create agent with message history
            agent_with_history = RunnableWithMessageHistory(
                self.agent_executor,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            # Configure with session
            config = {"configurable": {"session_id": session_id}}

            # Stream the response
            async for chunk in agent_with_history.astream(
                {"input": message}, config=config
            ):
                if "output" in chunk:
                    content = chunk["output"]
                    if content.strip():
                        yield content.replace("\n", "<br>")
                elif "intermediate_steps" in chunk:
                    # Optionally yield intermediate steps for debugging
                    steps = chunk["intermediate_steps"]
                    if steps:
                        for step in steps:
                            action, observation = step
                            yield f"<i>Using tool: {action.tool}</i><br>"

        except Exception as e:
            logger.error(f"Error in agent chat: {e}")
            yield f"I encountered an error while processing your request: {str(e)}"

    def chat_sync(
        self, message: str, session_id: str, category: Optional[int] = None
    ) -> Dict[str, Any]:
        """Synchronous chat method for non-streaming responses"""
        try:
            # Add category context if provided
            if category:
                message = f"[Category Level {category}] {message}"

            # Get chat history
            chat_history = self.get_session_history(session_id)

            # Create agent with message history
            agent_with_history = RunnableWithMessageHistory(
                self.agent_executor,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            # Configure with session
            config = {"configurable": {"session_id": session_id}}

            # Get response
            result = agent_with_history.invoke({"input": message}, config=config)

            return {
                "answer": result.get("output", "No response generated"),
                "intermediate_steps": result.get("intermediate_steps", []),
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"Error in sync agent chat: {e}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "intermediate_steps": [],
                "session_id": session_id,
            }

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get information about available tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": (
                    str(tool.args_schema.schema())
                    if hasattr(tool, "args_schema")
                    else "No schema"
                ),
            }
            for tool in self.tools
        ]

    def clear_session_history(self, session_id: str) -> bool:
        """Clear chat history for a session"""
        if session_id in self.chat_histories:
            self.chat_histories[session_id].clear()
            return True
        return False

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session history"""
        if session_id not in self.chat_histories:
            return {"message_count": 0, "messages": []}

        history = self.chat_histories[session_id]
        messages = []

        for message in history.messages:
            messages.append(
                {
                    "type": message.__class__.__name__,
                    "content": (
                        message.content[:100] + "..."
                        if len(message.content) > 100
                        else message.content
                    ),
                    "timestamp": getattr(message, "timestamp", None),
                }
            )

        return {"message_count": len(messages), "messages": messages}


class AgentManager:
    """Manager for multiple agent instances"""

    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service
        self.agents: Dict[str, BejoAgent] = {}
        self.default_agent_id = "default"

    def get_or_create_agent(
        self, agent_id: str = None, model_name: str = "gemini-2.0-flash"
    ) -> BejoAgent:
        """Get existing agent or create new one"""
        if agent_id is None:
            agent_id = self.default_agent_id

        if agent_id not in self.agents:
            self.agents[agent_id] = BejoAgent(self.retrieval_service, model_name)
            logger.info(f"Created new agent: {agent_id}")

        return self.agents[agent_id]

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent instance"""
        if agent_id in self.agents and agent_id != self.default_agent_id:
            del self.agents[agent_id]
            logger.info(f"Removed agent: {agent_id}")
            return True
        return False

    def list_agents(self) -> List[str]:
        """List all active agents"""
        return list(self.agents.keys())

    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents"""
        stats = {}
        for agent_id, agent in self.agents.items():
            stats[agent_id] = {
                "active_sessions": len(agent.chat_histories),
                "available_tools": len(agent.tools),
                "model": (
                    agent.llm.model_name
                    if hasattr(agent.llm, "model_name")
                    else "unknown"
                ),
            }
        return stats
