from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import logging

from .retrieval import RetrievalService

logger = logging.getLogger(__name__)


class DocumentRetrievalInput(BaseModel):
    """Input for document retrieval tool"""

    query: str = Field(description="The search query to find relevant documents")
    isa_level: int = Field(
        description="ISA-95 Level: 1=Field/Control, 2=Supervisory, 3=Planning, 4=Management",
        ge=1,
        le=4,
    )
    k: int = Field(
        default=3, description="Number of documents to retrieve", ge=1, le=10
    )
    strategy: str = Field(
        default="basic",
        description="Retrieval strategy: 'basic', 'hierarchical', or 'comprehensive'",
    )


class DocumentRetrievalTool(BaseTool):
    """Tool for retrieving relevant documents from ISA-95 knowledge base"""

    name: str = "document_retrieval"
    description: str = """
    Retrieve relevant documents from the ISA-95 based knowledge base.
    
    ISA-95 Levels:
    - Level 1 (Field & Control): Real-time control, sensors, actuators, basic automation
    - Level 2 (Supervisory): SCADA, HMI, batch control, recipe management  
    - Level 3 (Planning): Production scheduling, resource allocation, workflow management
    - Level 4 (Management): Business planning, KPIs, enterprise integration, strategic decisions
    
    Knowledge Access:
    - Level 1: Can access Level 1 knowledge only
    - Level 2: Can access Level 1,2 knowledge
    - Level 3: Can access Level 1,2,3 knowledge  
    - Level 4: Can access Level 1,2,3,4 knowledge (all)
    
    Parameters:
    - query: The search query (what information you're looking for)
    - isa_level: ISA-95 level to search from (determines knowledge scope)
    - k: Number of documents to retrieve (default: 3)
    - strategy: 'basic' for single collection, 'hierarchical' for weighted multi-level, 'comprehensive' for all levels
    """
    args_schema: type[BaseModel] = DocumentRetrievalInput

    def __init__(self, retrieval_service: RetrievalService):
        super().__init__()
        self.retrieval_service = retrieval_service

    def _run(
        self, query: str, isa_level: int, k: int = 3, strategy: str = "basic"
    ) -> str:
        """Execute the document retrieval"""
        try:
            # Get level information
            level_info = self.retrieval_service.get_available_knowledge_for_level(
                isa_level
            )
            level_name = level_info["level_name"]

            documents = self.retrieval_service.retrieve_documents(
                query=query, level=isa_level, strategy=strategy, search_kwargs={"k": k}
            )

            if not documents:
                return f"No relevant documents found for query: '{query}' at ISA-95 {level_name} level (Level {isa_level})"

            # Format the retrieved documents with ISA-95 context
            result = f"Found {len(documents)} relevant documents from ISA-95 {level_name} level:\n\n"
            result += f"Query: {query}\n"
            result += f"ISA-95 Level: {isa_level} ({level_name})\n"
            result += f"Available Knowledge Levels: {level_info['available_knowledge_levels']}\n"
            result += f"Strategy Used: {strategy}\n\n"

            for i, doc in enumerate(documents, 1):
                result += f"Document {i}:\n"
                result += f"Content: {doc.page_content[:500]}...\n"
                if doc.metadata:
                    # Extract knowledge level from metadata if available
                    source_level = "Unknown"
                    if "source" in doc.metadata:
                        source = doc.metadata["source"]
                        if "level-" in source:
                            source_level = source.split("level-")[-1].split("-")[0]

                    result += f"Source Knowledge Level: {source_level}\n"
                    result += f"Metadata: {doc.metadata}\n"
                result += "\n---\n\n"

            return result

        except Exception as e:
            logger.error(f"Error in ISA-95 document retrieval: {e}")
            return f"Error retrieving documents: {str(e)}"


class KnowledgeSearchInput(BaseModel):
    """Input for ISA-95 knowledge search tool"""

    query: str = Field(description="Search query for specific knowledge")
    isa_level: int = Field(description="ISA-95 Level to search from", ge=1, le=4)
    score_threshold: float = Field(
        default=0.7, description="Minimum similarity score", ge=0.0, le=1.0
    )
    focus_level: Optional[int] = Field(
        default=None,
        description="Focus on specific knowledge level within ISA scope",
        ge=1,
        le=4,
    )


class KnowledgeSearchTool(BaseTool):
    """Tool for searching ISA-95 knowledge with similarity scoring"""

    name: str = "knowledge_search"
    description: str = """
    Search for specific knowledge within ISA-95 hierarchy with similarity scoring.
    
    This tool allows you to search within the knowledge scope of a specific ISA-95 level:
    - Level 1 searches only Level 1 knowledge (Field & Control)
    - Level 2 searches Level 1,2 knowledge (Supervisory + Field)
    - Level 3 searches Level 1,2,3 knowledge (Planning + Supervisory + Field)
    - Level 4 searches Level 1,2,3,4 knowledge (Management + all lower levels)
    
    Parameters:
    - query: What specific knowledge you're looking for
    - isa_level: ISA-95 level context (determines knowledge scope)
    - score_threshold: Minimum similarity score (0.0-1.0, default: 0.7)
    - focus_level: Optional - focus on specific knowledge level within scope
    """
    args_schema: type[BaseModel] = KnowledgeSearchInput

    def __init__(self, retrieval_service: RetrievalService):
        super().__init__()
        self.retrieval_service = retrieval_service

    def _run(
        self,
        query: str,
        isa_level: int,
        score_threshold: float = 0.7,
        focus_level: Optional[int] = None,
    ) -> str:
        """Execute ISA-95 knowledge search with scoring"""
        try:
            # Get level information
            level_info = self.retrieval_service.get_available_knowledge_for_level(
                isa_level
            )
            level_name = level_info["level_name"]
            available_levels = level_info["available_knowledge_levels"]

            # If focus_level is specified, validate it's available
            if focus_level and focus_level not in available_levels:
                return f"Focus level {focus_level} is not available for ISA-95 {level_name} (Level {isa_level}). Available levels: {available_levels}"

            results = self.retrieval_service.search_similar_documents(
                query=query, level=isa_level, k=5, score_threshold=score_threshold
            )

            # Filter by focus_level if specified
            if focus_level:
                filtered_results = []
                for result in results:
                    # Try to determine source level from metadata
                    source_level = None
                    if "source" in result["metadata"]:
                        source = result["metadata"]["source"]
                        if f"level-{focus_level}" in source.lower():
                            source_level = focus_level

                    if source_level == focus_level:
                        filtered_results.append(result)

                results = filtered_results

            if not results:
                focus_text = (
                    f" with focus on Level {focus_level}" if focus_level else ""
                )
                return f"No knowledge found above threshold {score_threshold} for: '{query}' in ISA-95 {level_name}{focus_text}"

            # Format results with ISA-95 context
            focus_text = f" (Focused on Level {focus_level})" if focus_level else ""
            result = f"Found {len(results)} relevant knowledge items from ISA-95 {level_name}{focus_text}:\n\n"
            result += f"Query: {query}\n"
            result += f"ISA-95 Context: Level {isa_level} ({level_name})\n"
            result += f"Available Knowledge Levels: {available_levels}\n"
            result += f"Score Threshold: {score_threshold}\n\n"

            for i, item in enumerate(results, 1):
                score = item.get("similarity_score", 0.0)
                result += f"Knowledge {i} (Relevance: {score:.3f}):\n"
                result += f"{item['content'][:400]}...\n"

                # Extract source level info
                source_info = item["metadata"].get("source", "Unknown")
                if "level-" in source_info.lower():
                    try:
                        source_level = (
                            source_info.lower().split("level-")[1].split("-")[0]
                        )
                        result += f"Knowledge Level: {source_level}\n"
                    except:
                        pass

                result += f"Source: {source_info}\n\n"

            return result

        except Exception as e:
            logger.error(f"Error in ISA-95 knowledge search: {e}")
            return f"Error searching knowledge: {str(e)}"


class DocumentSummaryInput(BaseModel):
    """Input for document summary tool"""

    query: str = Field(description="Query to find documents to summarize")
    level: int = Field(description="Knowledge level", ge=1, le=4)
    max_docs: int = Field(
        default=5, description="Maximum documents to summarize", ge=1, le=10
    )


class DocumentSummaryTool(BaseTool):
    """Tool for getting document summaries"""

    name: str = "document_summary"
    description: str = """
    Get summaries of relevant documents for a topic.
    Use this when you need an overview of multiple documents about a subject.
    
    Parameters:
    - query: Topic or subject to find and summarize documents about
    - level: Knowledge level (1-4)
    - max_docs: Maximum number of documents to include in summary
    """
    args_schema: type[BaseModel] = DocumentSummaryInput

    def __init__(self, retrieval_service: RetrievalService):
        super().__init__()
        self.retrieval_service = retrieval_service

    def _run(self, query: str, level: int, max_docs: int = 5) -> str:
        """Execute document summarization"""
        try:
            documents = self.retrieval_service.retrieve_documents(
                query=query, level=level, search_kwargs={"k": max_docs}
            )

            if not documents:
                return f"No documents found to summarize for: '{query}'"

            # Create summary
            summary = f"Summary of {len(documents)} documents about '{query}':\n\n"

            key_points = []
            sources = []

            for i, doc in enumerate(documents, 1):
                # Extract key information
                content = doc.page_content[:300]
                source = doc.metadata.get("source", f"Document {i}")
                sources.append(source)

                # Simple key point extraction (first sentence or significant content)
                sentences = content.split(".")
                if sentences:
                    key_point = sentences[0].strip()
                    if len(key_point) > 20:  # Only meaningful sentences
                        key_points.append(f"• {key_point}")

            summary += "Key Points:\n"
            summary += "\n".join(key_points[:10])  # Top 10 key points
            summary += f"\n\nSources: {', '.join(set(sources))}"

            return summary

        except Exception as e:
            logger.error(f"Error in document summary: {e}")
            return f"Error creating summary: {str(e)}"


class CollectionStatsInput(BaseModel):
    """Input for collection stats tool"""

    detailed: bool = Field(
        default=False, description="Whether to include detailed statistics"
    )


class CollectionStatsTool(BaseTool):
    """Tool for getting collection statistics"""

    name: str = "collection_stats"
    description: str = """
    Get statistics about the knowledge base collections.
    Use this to understand what knowledge is available and collection health.
    
    Parameters:
    - detailed: Include detailed statistics (default: False)
    """
    args_schema: type[BaseModel] = CollectionStatsInput

    def __init__(self, retrieval_service: RetrievalService):
        super().__init__()
        self.retrieval_service = retrieval_service

    def _run(self, detailed: bool = False) -> str:
        """Get collection statistics"""
        try:
            stats = self.retrieval_service.get_collection_stats()

            result = "Knowledge Base Statistics:\n\n"

            total_documents = 0
            healthy_collections = 0

            for collection_name, collection_stats in stats.items():
                if "error" in collection_stats:
                    result += (
                        f"❌ {collection_name}: Error - {collection_stats['error']}\n"
                    )
                else:
                    points_count = collection_stats.get("points_count", 0)
                    status = collection_stats.get("status", "unknown")

                    total_documents += points_count
                    if status == "green":
                        healthy_collections += 1

                    result += (
                        f"✅ {collection_name}: {points_count} documents ({status})\n"
                    )

                    if detailed:
                        vectors_count = collection_stats.get("vectors_count", 0)
                        result += f"   - Vectors: {vectors_count}\n"

            result += f"\nSummary:\n"
            result += f"- Total Documents: {total_documents}\n"
            result += f"- Healthy Collections: {healthy_collections}/{len(stats)}\n"

            return result

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return f"Error retrieving statistics: {str(e)}"


def create_retrieval_tools(retrieval_service: RetrievalService) -> List[BaseTool]:
    """Create all retrieval-related tools"""
    return [
        DocumentRetrievalTool(retrieval_service),
        KnowledgeSearchTool(retrieval_service),
        DocumentSummaryTool(retrieval_service),
        CollectionStatsTool(retrieval_service),
    ]
