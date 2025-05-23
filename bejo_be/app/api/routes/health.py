from fastapi import APIRouter, Depends
import logging
from ...services.dependencies import get_agent_manager, get_retrieval_service
from ...core.agent import AgentManager
from ...core.retrieval import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "message": "BEJO Agentic RAG API is running"}


@router.get("/detailed")
async def detailed_health_check(
    agent_manager: AgentManager = Depends(get_agent_manager),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Detailed health check with component status"""
    try:
        # Check retrieval service
        collection_stats = retrieval_service.get_collection_stats()
        healthy_collections = sum(
            1 for stats in collection_stats.values() if "error" not in stats
        )

        # Check agent manager
        agent_stats = agent_manager.get_agent_stats()

        return {
            "status": "healthy",
            "components": {
                "retrieval_service": {
                    "status": "healthy",
                    "collections": len(collection_stats),
                    "healthy_collections": healthy_collections,
                },
                "agent_manager": {
                    "status": "healthy",
                    "active_agents": len(agent_stats),
                    "agents": agent_stats,
                },
            },
            "message": "All components are healthy",
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Health check failed",
        }
