"""
MindFu Health Check API
"""
import logging
import time
from typing import Dict

import httpx
import redis
from fastapi import APIRouter
from qdrant_client import QdrantClient

from ..core.config import get_settings
from ..models.schemas import HealthResponse, ServiceStatus

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

VERSION = "0.1.0"


async def check_llm() -> ServiceStatus:
    """Check LLM service health."""
    settings = get_settings()
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.llm_base_url.replace('/v1', '')}/health")
            latency = (time.time() - start) * 1000

            return ServiceStatus(
                name="llm",
                healthy=response.status_code == 200,
                latency_ms=latency,
            )
    except Exception as e:
        return ServiceStatus(
            name="llm",
            healthy=False,
            error=str(e),
        )


def check_qdrant() -> ServiceStatus:
    """Check Qdrant health."""
    settings = get_settings()
    start = time.time()

    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=5)
        client.get_collections()
        latency = (time.time() - start) * 1000

        return ServiceStatus(
            name="qdrant",
            healthy=True,
            latency_ms=latency,
        )
    except Exception as e:
        return ServiceStatus(
            name="qdrant",
            healthy=False,
            error=str(e),
        )


def check_redis() -> ServiceStatus:
    """Check Redis health."""
    settings = get_settings()
    start = time.time()

    try:
        r = redis.Redis(host=settings.redis_host, port=settings.redis_port, socket_timeout=5)
        r.ping()
        latency = (time.time() - start) * 1000

        return ServiceStatus(
            name="redis",
            healthy=True,
            latency_ms=latency,
        )
    except Exception as e:
        return ServiceStatus(
            name="redis",
            healthy=False,
            error=str(e),
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check health of all services.

    Returns overall health status and individual service statuses.
    """
    # Check all services
    llm_status = await check_llm()
    qdrant_status = check_qdrant()
    redis_status = check_redis()

    services = {
        "llm": llm_status.healthy,
        "qdrant": qdrant_status.healthy,
        "redis": redis_status.healthy,
    }

    # Overall health - RAG can work without LLM (for document indexing)
    # but needs Qdrant
    all_healthy = all(services.values())
    critical_healthy = services["qdrant"]  # Qdrant is critical

    return HealthResponse(
        status="healthy" if critical_healthy else "unhealthy",
        version=VERSION,
        services=services,
    )


@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    qdrant_status = check_qdrant()

    if not qdrant_status.healthy:
        return {"status": "not ready", "reason": "qdrant unavailable"}

    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get("/health/detailed")
async def detailed_health():
    """Get detailed health information for all services."""
    llm_status = await check_llm()
    qdrant_status = check_qdrant()
    redis_status = check_redis()

    return {
        "version": VERSION,
        "services": [
            llm_status.model_dump(),
            qdrant_status.model_dump(),
            redis_status.model_dump(),
        ],
    }
