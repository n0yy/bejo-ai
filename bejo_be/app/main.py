import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config.settings import get_settings
from .api.routes import chat, knowledge, health
from .services.dependencies import get_retrieval_service
from .utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger = logging.getLogger(__name__)
    logger.info("Starting BEJO Agentic RAG API...")

    try:
        # Initialize retrieval service and collections
        retrieval_service = get_retrieval_service()
        logger.info("Retrieval service initialized")

        # Ensure basic collections exist
        retrieval_service._ensure_collection_exists("bejo-knowledge-level-1")
        logger.info("Collections initialized")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down BEJO Agentic RAG API...")


def create_app() -> FastAPI:
    """Create FastAPI application"""
    settings = get_settings()

    # Setup logging
    setup_logging(debug=settings.debug)

    # Create app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Agentic RAG API for intelligent document retrieval and chat",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )

    # Include routers
    app.include_router(chat.router)
    app.include_router(knowledge.router)
    app.include_router(health.router)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )
