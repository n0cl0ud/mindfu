"""
MindFu Chat API - OpenAI-Compatible Endpoints
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..core.config import get_settings
from ..core.rag_chain import get_rag_chain
from ..models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    ChatUsage,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


async def log_conversation_async(
    model: str,
    messages: list,
    response_content: str | None,
    response_tool_calls: list | None,
    finish_reason: str | None,
    usage: dict | None,
    rag_context: dict | None,
):
    """Log conversation in background (non-blocking)."""
    try:
        settings = get_settings()
        if not settings.log_conversations:
            return

        from ..core.database import get_conversation_logger

        conv_logger = get_conversation_logger()
        await conv_logger.log_conversation(
            model=model,
            messages=messages,
            response_content=response_content,
            response_tool_calls=response_tool_calls,
            finish_reason=finish_reason,
            tokens_prompt=usage.get("prompt_tokens") if usage else None,
            tokens_completion=usage.get("completion_tokens") if usage else None,
            tokens_total=usage.get("total_tokens") if usage else None,
            rag_contexts_used=rag_context.get("contexts_used", 0) if rag_context else 0,
            rag_sources=rag_context.get("sources") if rag_context else None,
        )
    except Exception as e:
        logger.warning(f"Failed to log conversation: {e}")


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with optional RAG augmentation.

    This endpoint is compatible with:
    - Claude CLI (claude-code)
    - OpenAI Python SDK
    - LangChain
    - Any OpenAI-compatible client
    """
    try:
        rag_chain = get_rag_chain()

        # Convert messages to dict format (exclude None values for llama.cpp compatibility)
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

        if request.stream:
            return StreamingResponse(
                stream_response(rag_chain, messages, request),
                media_type="text/event-stream",
            )

        # Non-streaming response
        result = await rag_chain.query(
            messages=messages,
            collection=request.collection,
            use_rag=request.use_rag,
            stream=False,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            tools=[t.model_dump() for t in request.tools] if request.tools else None,
            tool_choice=request.tool_choice,
        )

        # Format response
        response = ChatCompletionResponse(
            id=result.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
            created=result.get("created", int(datetime.now().timestamp())),
            model=result.get("model", request.model),
            choices=[
                ChatChoice(
                    index=i,
                    message=ChatMessage(
                        role=choice["message"]["role"],
                        content=choice["message"]["content"],
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
                for i, choice in enumerate(result.get("choices", []))
            ],
            usage=ChatUsage(**result["usage"]) if result.get("usage") else None,
        )

        # Log conversation in background
        response_content = None
        response_tool_calls = None
        finish_reason = None

        if result.get("choices"):
            first_choice = result["choices"][0]
            message = first_choice.get("message", {})
            response_content = message.get("content")
            response_tool_calls = message.get("tool_calls")
            finish_reason = first_choice.get("finish_reason")

        asyncio.create_task(
            log_conversation_async(
                model=request.model,
                messages=messages,
                response_content=response_content,
                response_tool_calls=response_tool_calls,
                finish_reason=finish_reason,
                usage=result.get("usage"),
                rag_context=result.get("_rag_context"),
            )
        )

        # Add RAG context if available
        if "_rag_context" in result:
            response_dict = response.model_dump()
            response_dict["_rag_context"] = result["_rag_context"]
            return response_dict

        return response

    except Exception as e:
        logger.exception("Chat completion error")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(
    rag_chain,
    messages: list,
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Generate streaming response."""
    try:
        async for chunk in await rag_chain.query(
            messages=messages,
            collection=request.collection,
            use_rag=request.use_rag,
            stream=True,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            tools=[t.model_dump() for t in request.tools] if request.tools else None,
            tool_choice=request.tool_choice,
        ):
            yield chunk

    except Exception as e:
        logger.exception("Streaming error")
        error_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@router.get("/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "devstral-small-2",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "mindfu",
            }
        ],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model info (OpenAI-compatible)."""
    return {
        "id": model_id,
        "object": "model",
        "created": int(datetime.now().timestamp()),
        "owned_by": "mindfu",
    }
