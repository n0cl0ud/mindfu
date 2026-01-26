"""
MindFu Conversations API - Logging, Export, and Training Data Management
"""
import json
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.database import get_conversation_logger
from ..core.rag_chain import get_rag_chain

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["conversations"])


# =============================================================================
# Schemas
# =============================================================================


class ConversationResponse(BaseModel):
    """Single conversation response."""

    id: str
    created_at: datetime
    model: str
    messages: List[dict]
    response_content: Optional[str]
    response_tool_calls: Optional[List[dict]]
    finish_reason: Optional[str]
    tokens_total: Optional[int]
    rag_contexts_used: int
    training_quality: Optional[str]


class ConversationListResponse(BaseModel):
    """List of conversations."""

    conversations: List[ConversationResponse]
    total: int
    offset: int
    limit: int


class QualityUpdateRequest(BaseModel):
    """Request to update conversation quality."""

    quality: str = Field(..., pattern="^(good|bad|skip)$")


class SaveToRAGRequest(BaseModel):
    """Request to save a conversation to RAG."""

    conversation_id: str
    collection: Optional[str] = None
    include_context: bool = Field(default=True, description="Include original question as context")


class ExportRequest(BaseModel):
    """Request to export conversations for training."""

    quality_filter: Optional[str] = Field(default="good", pattern="^(good|bad|skip)$")
    limit: Optional[int] = Field(default=None, ge=1)
    mark_exported: bool = Field(default=True)
    format: str = Field(default="alpaca", pattern="^(alpaca|chatml|jsonl)$")


class StatsResponse(BaseModel):
    """Conversation statistics."""

    total_conversations: int
    by_quality: dict
    exported: int
    total_tokens: int


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    model: Optional[str] = None,
    unexported_only: bool = False,
):
    """List logged conversations."""
    try:
        conv_logger = get_conversation_logger()
        conversations = await conv_logger.get_conversations(
            limit=limit,
            offset=offset,
            model=model,
            unexported_only=unexported_only,
        )

        return ConversationListResponse(
            conversations=[
                ConversationResponse(
                    id=c.id,
                    created_at=c.created_at,
                    model=c.model,
                    messages=c.messages,
                    response_content=c.response_content,
                    response_tool_calls=c.response_tool_calls,
                    finish_reason=c.finish_reason,
                    tokens_total=c.tokens_total,
                    rag_contexts_used=c.rag_contexts_used,
                    training_quality=c.training_quality,
                )
                for c in conversations
            ],
            total=len(conversations),
            offset=offset,
            limit=limit,
        )

    except Exception as e:
        logger.exception("List conversations error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/stats", response_model=StatsResponse)
async def get_stats():
    """Get conversation statistics."""
    try:
        conv_logger = get_conversation_logger()
        stats = await conv_logger.get_stats()
        return StatsResponse(**stats)

    except Exception as e:
        logger.exception("Get stats error")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/conversations/{conversation_id}/quality")
async def update_quality(conversation_id: str, request: QualityUpdateRequest):
    """Update the training quality rating for a conversation."""
    try:
        conv_logger = get_conversation_logger()
        success = await conv_logger.set_quality(conversation_id, request.quality)

        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"message": f"Quality set to '{request.quality}'", "conversation_id": conversation_id}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Update quality error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/save-to-rag")
async def save_to_rag(request: SaveToRAGRequest):
    """
    Save a conversation response to the RAG knowledge base.

    This allows you to build up a knowledge base from good responses.
    """
    try:
        conv_logger = get_conversation_logger()
        rag_chain = get_rag_chain()

        # Get the conversation
        conversations = await conv_logger.get_conversations(limit=1000)
        conversation = next((c for c in conversations if c.id == request.conversation_id), None)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if not conversation.response_content:
            raise HTTPException(status_code=400, detail="Conversation has no response content")

        # Build content to save
        content_parts = []

        if request.include_context:
            # Add the question/instruction
            user_messages = [m for m in conversation.messages if m.get("role") == "user"]
            if user_messages:
                content_parts.append(f"Question: {user_messages[-1].get('content', '')}")
                content_parts.append("")

        content_parts.append(f"Answer: {conversation.response_content}")

        content = "\n".join(content_parts)

        # Save to Qdrant
        doc_id = rag_chain.add_document(
            content=content,
            metadata={
                "source": f"conversation:{conversation.id}",
                "type": "conversation",
                "model": conversation.model,
                "created_at": conversation.created_at.isoformat(),
            },
            collection=request.collection,
        )

        return {
            "message": "Conversation saved to RAG",
            "document_id": doc_id,
            "conversation_id": conversation.id,
            "collection": request.collection or "documents",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Save to RAG error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/export")
async def export_conversations(request: ExportRequest):
    """
    Export conversations for fine-tuning.

    Returns training data in the specified format.
    """
    try:
        conv_logger = get_conversation_logger()

        # Get training data
        training_data = await conv_logger.export_for_training(
            quality_filter=request.quality_filter,
            limit=request.limit,
        )

        if not training_data:
            raise HTTPException(status_code=404, detail="No conversations found matching criteria")

        # Mark as exported if requested
        if request.mark_exported:
            conversation_ids = [d["metadata"]["conversation_id"] for d in training_data]
            await conv_logger.mark_exported(conversation_ids)

        # Format output
        if request.format == "alpaca":
            # Alpaca format (instruction, input, output)
            output = [
                {
                    "instruction": d["instruction"],
                    "input": d["input"],
                    "output": d["output"],
                }
                for d in training_data
            ]
        elif request.format == "chatml":
            # ChatML format (messages array)
            output = []
            for d in training_data:
                messages = [{"role": "user", "content": d["instruction"]}]
                if d["input"]:
                    messages.insert(0, {"role": "system", "content": d["input"]})
                messages.append({"role": "assistant", "content": d["output"]})
                output.append({"messages": messages})
        else:
            # JSONL - one JSON object per line
            output = training_data

        # Return as downloadable JSON
        content = json.dumps(output, indent=2, ensure_ascii=False)

        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Export error")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation from the log."""
    try:
        from sqlalchemy import delete

        from ..core.database import Conversation, get_session

        async with get_session() as session:
            stmt = delete(Conversation).where(Conversation.id == conversation_id)
            result = await session.execute(stmt)

            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Conversation not found")

            return {"message": "Conversation deleted", "conversation_id": conversation_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Delete conversation error")
        raise HTTPException(status_code=500, detail=str(e))
