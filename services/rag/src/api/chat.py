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


async def fake_stream_response(result: dict) -> AsyncGenerator[str, None]:
    """
    Convert a non-streaming response to SSE format.
    Used when we force non-streaming due to vLLM tool call bugs but client expects streaming.
    Mimics real streaming by sending role, content, and finish_reason in separate chunks.
    """
    if not result.get("choices"):
        yield "data: [DONE]\n\n"
        return

    choice = result["choices"][0]
    message = choice.get("message", {})
    chunk_id = result.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")
    created = result.get("created", int(datetime.now().timestamp()))
    model = result.get("model", "")

    # Chunk 1: role
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'logprobs': None, 'finish_reason': None}]})}\n\n"

    # Chunk 2: content (if any)
    content = message.get("content")
    if content:
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': content}, 'logprobs': None, 'finish_reason': None}]})}\n\n"

    # Chunk 3: tool_calls (if any)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        # Convert to streaming format with index
        streaming_tool_calls = [{"index": i, **tc} for i, tc in enumerate(tool_calls)]
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'tool_calls': streaming_tool_calls}, 'logprobs': None, 'finish_reason': None}]})}\n\n"

    # Chunk 4: finish_reason + usage (combined as expected by Vibe/Mistral clients)
    final_chunk = {
        'id': chunk_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': model,
        'choices': [{'index': 0, 'delta': {'content': ''}, 'logprobs': None, 'finish_reason': choice.get('finish_reason', 'stop')}]
    }
    usage = result.get("usage")
    if usage:
        final_chunk['usage'] = usage
    yield f"data: {json.dumps(final_chunk)}\n\n"

    # Send the [DONE] marker
    yield "data: [DONE]\n\n"


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

        # Convert messages to dict format
        # Keep content field (even if None) for vLLM tool call compatibility
        # But exclude other None fields to avoid 400 errors
        def convert_message(msg):
            d = msg.model_dump(exclude_none=True)
            # Preserve content:null for assistant messages with tool_calls
            if msg.role == "assistant" and msg.tool_calls and msg.content is None:
                d["content"] = None
            return d
        messages = [convert_message(msg) for msg in request.messages]

        # WORKAROUND: Disable streaming when tools are present
        # vLLM's Mistral tool parser has a bug with streaming tool calls
        # See: https://github.com/vllm-project/vllm/issues/17585
        # Can be disabled via FORCE_NO_STREAM_WITH_TOOLS=false for models with working streaming (e.g., Nemotron)
        settings = get_settings()
        force_no_stream = bool(request.tools) and settings.force_no_stream_with_tools
        if force_no_stream and request.stream:
            logger.info("Forcing non-streaming mode due to tool calls (vLLM bug workaround)")

        # WORKAROUND: Disable RAG when tools are present
        # RAG context can confuse models during tool calling tasks
        use_rag = request.use_rag
        if request.tools:
            use_rag = False
            logger.info("Disabling RAG for tool call request")

        if request.stream and not force_no_stream:
            return StreamingResponse(
                stream_response(rag_chain, messages, request),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )

        # Non-streaming response (or forced non-streaming for tool calls)
        result = await rag_chain.query(
            messages=messages,
            collection=request.collection,
            use_rag=use_rag,
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

        # If client wanted streaming but we forced non-streaming, return non-streaming response
        # NOTE: fake_stream_response was causing issues with Vibe parsing tool_calls
        # Vibe should handle non-streaming responses fine
        if force_no_stream and request.stream:
            # Return as non-streaming JSON response instead of fake SSE
            return result

        # Return raw LLM response with RAG context (preserves all vLLM fields)
        return result

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
            stream_options=request.stream_options.model_dump() if request.stream_options else None,
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
