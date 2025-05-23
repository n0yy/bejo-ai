import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.api import chat, knowledge, health
from app.services.vectors import VectorService

# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting BEJO RAG API...")

    # Initialize vector service and ensure collections exist
    vector_service = VectorService()

    # Ensure knowledge collections exist
    for category, collection_name in settings.CATEGORY_COLLECTIONS.items():
        await vector_service.ensure_collection_exists(collection_name)

    # Ensure chat history collection exists
    await vector_service.ensure_collection_exists(settings.CHAT_HISTORY_COLLECTION)

    logger.info("Application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down BEJO RAG API...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(health.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
