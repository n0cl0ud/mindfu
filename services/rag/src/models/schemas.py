"""
MindFu API Schemas
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# OpenAI-Compatible Chat Schemas
# =============================================================================


class ToolCall(BaseModel):
    """Tool call in a message."""

    id: str
    type: str = "function"
    function: Dict[str, Any]


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response messages


class ToolFunction(BaseModel):
    """Tool function definition."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """Tool definition."""

    type: str = "function"
    function: ToolFunction


class StreamOptions(BaseModel):
    """Stream options for including usage in streaming responses."""

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "devstral-small-2"
    messages: List[ChatMessage] = Field(max_length=256)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    top_p: float = Field(default=1.0, ge=0, le=1)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    stop: Optional[List[str]] = None

    # Tool calling support
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Any] = None  # Can be "auto", "none", or specific tool

    # RAG-specific options
    use_rag: bool = Field(default=True, description="Enable RAG context retrieval")
    collection: Optional[str] = Field(default=None, description="Qdrant collection to search")
    top_k: Optional[int] = Field(default=None, description="Number of context chunks to retrieve")


class ChatChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[ChatUsage] = None

    # RAG metadata (optional)
    rag_context: Optional[Dict[str, Any]] = Field(default=None, alias="_rag_context")


# =============================================================================
# Document Schemas
# =============================================================================


class DocumentChunk(BaseModel):
    """A chunk of a document."""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUploadRequest(BaseModel):
    """Request to upload a document."""

    content: str = Field(max_length=10_000_000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection: Optional[str] = None
    chunk: bool = Field(default=True, description="Whether to chunk the document")


class DocumentUploadResponse(BaseModel):
    """Response from document upload."""

    document_ids: List[str]
    chunks_created: int
    collection: str


class DocumentQueryRequest(BaseModel):
    """Request to query documents."""

    query: str = Field(max_length=10_000)
    collection: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=100)


class DocumentQueryResult(BaseModel):
    """A single query result."""

    content: str
    metadata: Dict[str, Any]
    score: float


class DocumentQueryResponse(BaseModel):
    """Response from document query."""

    results: List[DocumentQueryResult]
    query: str
    collection: str


# =============================================================================
# Collection Schemas
# =============================================================================


class CollectionInfo(BaseModel):
    """Information about a collection."""

    name: str
    vectors_count: int
    points_count: int


class CollectionCreateRequest(BaseModel):
    """Request to create a collection."""

    name: str = Field(min_length=1, max_length=255, pattern=r"^[a-zA-Z0-9_-]+$")
    vector_size: Optional[int] = None


class CollectionListResponse(BaseModel):
    """Response listing collections."""

    collections: List[CollectionInfo]


# =============================================================================
# Conversation Logging Schemas
# =============================================================================


class ConversationLog(BaseModel):
    """Logged conversation for training."""

    id: str
    messages: List[ChatMessage]
    response: ChatMessage
    model: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Health & Status Schemas
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"]
    version: str
    services: Dict[str, bool]


class ServiceStatus(BaseModel):
    """Status of a service."""

    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
