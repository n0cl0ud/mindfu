"""
MindFu Metrics API - Unified Prometheus metrics endpoint
"""
import logging
import time
from typing import Optional

import httpx
from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
    REGISTRY,
)

from ..core.config import get_settings
from ..core.rag_chain import get_rag_chain

logger = logging.getLogger(__name__)
router = APIRouter(tags=["metrics"])

# =============================================================================
# RAG Metrics
# =============================================================================

RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total RAG requests",
    ["endpoint", "status"]
)

RAG_REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "RAG request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

RAG_CONTEXT_CHUNKS = Histogram(
    "rag_context_chunks",
    "Number of context chunks retrieved per request",
    buckets=[0, 1, 2, 3, 4, 5, 10, 20]
)

RAG_DOCUMENTS_INGESTED = Counter(
    "rag_documents_ingested_total",
    "Total documents ingested",
    ["action"]  # created, updated, skipped
)

# =============================================================================
# Qdrant Metrics (updated on each /metrics call)
# =============================================================================

QDRANT_COLLECTIONS = Gauge(
    "qdrant_collections_total",
    "Total number of Qdrant collections"
)

QDRANT_VECTORS = Gauge(
    "qdrant_vectors_total",
    "Total vectors in Qdrant",
    ["collection"]
)

QDRANT_POINTS = Gauge(
    "qdrant_points_total",
    "Total points in Qdrant",
    ["collection"]
)

QDRANT_STATUS = Gauge(
    "qdrant_collection_status",
    "Collection status (1=green, 0=other)",
    ["collection"]
)


def update_qdrant_metrics():
    """Update Qdrant metrics from current state."""
    try:
        rag_chain = get_rag_chain()
        collections = rag_chain.qdrant.get_collections().collections

        QDRANT_COLLECTIONS.set(len(collections))

        for col in collections:
            try:
                info = rag_chain.qdrant.get_collection(col.name)
                QDRANT_VECTORS.labels(collection=col.name).set(info.vectors_count or 0)
                QDRANT_POINTS.labels(collection=col.name).set(info.points_count or 0)
                QDRANT_STATUS.labels(collection=col.name).set(1 if info.status.value == "green" else 0)
            except Exception as e:
                logger.warning(f"Failed to get stats for collection {col.name}: {e}")
    except Exception as e:
        logger.warning(f"Failed to update Qdrant metrics: {e}")


async def fetch_vllm_metrics() -> Optional[str]:
    """Fetch metrics from vLLM."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.llm_base_url}/metrics")
            if response.status_code == 200:
                return response.text
    except Exception as e:
        logger.warning(f"Failed to fetch vLLM metrics: {e}")
    return None


@router.get("/metrics")
async def metrics():
    """
    Unified Prometheus metrics endpoint.

    Includes:
    - RAG service metrics
    - Qdrant metrics
    - vLLM metrics (forwarded)
    """
    # Update Qdrant metrics
    update_qdrant_metrics()

    # Generate RAG metrics
    rag_metrics = generate_latest(REGISTRY).decode("utf-8")

    # Fetch vLLM metrics
    vllm_metrics = await fetch_vllm_metrics()

    # Combine all metrics
    all_metrics = rag_metrics
    if vllm_metrics:
        # Add vLLM metrics with prefix comment
        all_metrics += "\n# vLLM metrics (forwarded)\n"
        all_metrics += vllm_metrics

    return Response(content=all_metrics, media_type=CONTENT_TYPE_LATEST)


# =============================================================================
# Helper functions to record metrics (called from other modules)
# =============================================================================

def record_request(endpoint: str, status: str, latency: float):
    """Record a RAG request."""
    RAG_REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()
    RAG_REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)


def record_context_chunks(count: int):
    """Record number of context chunks retrieved."""
    RAG_CONTEXT_CHUNKS.observe(count)


def record_document_ingested(action: str):
    """Record a document ingestion."""
    RAG_DOCUMENTS_INGESTED.labels(action=action).inc()
