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
    Streams tool_calls in OpenAI-compatible incremental format.
    """
    if not result.get("choices"):
        yield "data: [DONE]\n\n"
        return

    choice = result["choices"][0]
    message = choice.get("message", {})
    chunk_id = result.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")
    created = result.get("created", int(datetime.now().timestamp()))
    model = result.get("model", "")
    usage = result.get("usage")

    content = message.get("content")
    tool_calls = message.get("tool_calls")

    # Build first delta with role (and tool_calls if present - must be together for Vibe)
    first_delta = {"role": "assistant"}

    if tool_calls:
        # Send role + complete tool_calls in same chunk (Vibe expects them together)
        streaming_tool_calls = []
        for i, tc in enumerate(tool_calls):
            args = tc.get("function", {}).get("arguments", "")
            logger.info(f"DEBUG fake_stream tool_call[{i}]: id={tc.get('id')}, name={tc.get('function', {}).get('name')}, args_len={len(args)}, args_preview={args[:200] if args else 'empty'}")
            streaming_tool_calls.append({
                "index": i,
                "id": tc.get("id", ""),
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": args
                }
            })
        first_delta["tool_calls"] = streaming_tool_calls
        chunk_data = {'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': first_delta, 'logprobs': None, 'finish_reason': None}]}
        chunk_str = json.dumps(chunk_data)
        logger.info(f"DEBUG fake_stream yielding role+tool_calls chunk: {chunk_str[:500]}")
        yield f"data: {chunk_str}\n\n"
    else:
        # No tool calls - send role, then content
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': first_delta, 'logprobs': None, 'finish_reason': None}]})}\n\n"

        if content:
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': content}, 'logprobs': None, 'finish_reason': None}]})}\n\n"

    # Final chunk: finish_reason + usage
    final_chunk = {
        'id': chunk_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': model,
        'choices': [{'index': 0, 'delta': {}, 'logprobs': None, 'finish_reason': choice.get('finish_reason', 'stop')}]
    }
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

        # If client wanted streaming but we forced non-streaming for tool calls,
        # return non-streaming JSON directly. Vibe can handle non-streaming responses
        # and cannot properly accumulate streamed tool_call arguments.
        # See: https://github.com/mistralai/mistral-vibe/issues/252
        if force_no_stream and request.stream:
            logger.info("Returning non-streaming JSON for tool call (Vibe can't accumulate streamed tool_calls)")

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
