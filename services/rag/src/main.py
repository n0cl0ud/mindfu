"""
MindFu RAG Service - Main Application
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import chat, conversations, documents, health
from .core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting MindFu RAG Service...")

    # Initialize services on startup
    settings = get_settings()
    logger.info(f"LLM Base URL: {settings.llm_base_url}")
    logger.info(f"Qdrant URL: {settings.qdrant_url}")
    logger.info(f"Embedding Model: {settings.embedding_model}")

    # Pre-load embedding model
    from .core.embeddings import get_embedding_service
    embedding_service = get_embedding_service()
    logger.info(f"Embedding dimension: {embedding_service.dimension}")

    # Initialize RAG chain (creates default collection if needed)
    from .core.rag_chain import get_rag_chain
    get_rag_chain()
    logger.info("RAG chain initialized")

    # Initialize database (create tables if needed)
    if settings.log_conversations:
        from .core.database import init_database
        await init_database()
        logger.info("Conversation logging enabled")

    yield

    logger.info("Shutting down MindFu RAG Service...")


# Create FastAPI app
app = FastAPI(
    title="MindFu RAG Service",
    description="OpenAI-compatible API with RAG augmentation for local LLM inference",
    version="0.1.0",
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
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(conversations.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MindFu RAG Service",
        "version": "0.1.0",
        "description": "OpenAI-compatible API with RAG augmentation",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "documents": "/v1/documents",
            "collections": "/v1/collections",
            "conversations": "/v1/conversations",
            "health": "/health",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
