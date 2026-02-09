"""
MindFu Chat API - OpenAI-Compatible Endpoints
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

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
    Convert a non-streaming response to OpenAI-exact SSE format.
    Used when we force non-streaming due to vLLM Mistral streaming bugs but client expects SSE.

    Follows the exact OpenAI streaming format for tool calls:
    1. role + tool_call init (id, type, name, empty arguments)
    2. tool_call arguments (streamed incrementally in small chunks)
    3. finish_reason + usage

    Known vLLM issues with Mistral streaming tool calls:
    - https://github.com/vllm-project/vllm/issues/17585
    - https://github.com/vllm-project/vllm/issues/20028
    - https://github.com/vllm-project/vllm/issues/29968
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

    def make_chunk(delta, finish_reason=None):
        return json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'logprobs': None, 'finish_reason': finish_reason}]})

    if tool_calls:
        # OpenAI format: chunk 1 = role + tool_call init (name, id, empty arguments)
        init_tool_calls = []
        for i, tc in enumerate(tool_calls):
            init_tool_calls.append({
                "index": i,
                "id": tc.get("id", ""),
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": ""
                }
            })
        yield f"data: {make_chunk({'role': 'assistant', 'tool_calls': init_tool_calls})}\n\n"

        # OpenAI format: chunk 2..N = arguments streamed incrementally
        # Real OpenAI streaming sends arguments in small chunks, not all at once.
        # Large tool calls (e.g. write_file with full file content) need chunking
        # so clients like Vibe can parse them progressively.
        ARG_CHUNK_SIZE = 512
        for i, tc in enumerate(tool_calls):
            args = tc.get("function", {}).get("arguments", "")
            if args:
                for offset in range(0, len(args), ARG_CHUNK_SIZE):
                    chunk_args = args[offset:offset + ARG_CHUNK_SIZE]
                    yield f"data: {make_chunk({'tool_calls': [{'index': i, 'function': {'arguments': chunk_args}}]})}\n\n"
    else:
        # Text response: role, then content
        yield f"data: {make_chunk({'role': 'assistant'})}\n\n"
        if content:
            yield f"data: {make_chunk({'content': content})}\n\n"

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

        # Check if we should force non-streaming for tool calls
        # This was a workaround for vLLM Mistral parser bugs, but may cause issues with Vibe
        # which expects SSE streaming format. Disable via FORCE_NO_STREAM_WITH_TOOLS=false
        settings = get_settings()
        force_no_stream = bool(request.tools) and settings.force_no_stream_with_tools
        if force_no_stream and request.stream:
            logger.info("Forcing non-streaming mode due to tool calls (vLLM bug workaround)")

        # Log streaming mode for debugging
        logger.info(f"Request: stream={request.stream}, tools={bool(request.tools)}, force_no_stream={force_no_stream}")

        # WORKAROUND: Disable RAG when tools are present
        # RAG context can confuse models during tool calling tasks
        use_rag = request.use_rag
        if request.tools:
            use_rag = False
            logger.info("Disabling RAG for tool call request")

        if request.stream and not force_no_stream:
            return StreamingResponse(
                stream_response(rag_chain, messages, request, use_rag),
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

            # Log tool call details for debugging truncation issues
            if response_tool_calls:
                for i, tc in enumerate(response_tool_calls):
                    args = tc.get("function", {}).get("arguments", "")
                    args_len = len(args) if isinstance(args, str) else -1
                    logger.info(f"Tool call {i}: {tc.get('function', {}).get('name', '?')} args_len={args_len} finish={finish_reason}")

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
        # wrap in fake_stream_response to return proper SSE format.
        # vLLM Mistral streaming tool calls are broken (args get corrupted).
        if force_no_stream and request.stream:
            logger.info("Wrapping non-streaming tool call response as SSE (vLLM Mistral streaming bug)")
            return StreamingResponse(
                fake_stream_response(result),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )

        # Return raw LLM response (preserves all vLLM fields)
        return result

    except httpx.HTTPStatusError as e:
        # Forward vLLM errors in OpenAI-compatible format
        status = e.response.status_code
        try:
            body = e.response.json()
            vllm_msg = body.get("message", "") or body.get("error", {}).get("message", "")
        except Exception:
            vllm_msg = e.response.text[:500]

        # Detect context length exceeded
        is_context_error = any(k in vllm_msg.lower() for k in [
            "maximum context length", "prompt is too long",
            "input tokens", "exceed", "too many tokens",
        ])
        # Also detect by input size (rough estimate: 4 chars per token)
        approx_tokens = sum(len(json.dumps(m)) for m in messages) // 4
        if not is_context_error and status >= 400:
            is_context_error = approx_tokens > 60000  # ~75% of 81920

        if is_context_error:
            logger.warning(f"Context length exceeded: ~{approx_tokens} tokens estimated")
            return JSONResponse(status_code=400, content={
                "error": {
                    "message": f"This model's maximum context length is {settings.max_model_len} tokens. "
                               f"Your messages resulted in approximately {approx_tokens} tokens. "
                               f"Please reduce the length of the messages.",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "context_length_exceeded",
                }
            })

        logger.exception("LLM backend error")
        return JSONResponse(status_code=status, content={
            "error": {
                "message": vllm_msg or str(e),
                "type": "server_error",
                "code": None,
            }
        })

    except httpx.TimeoutException:
        approx_tokens = sum(len(json.dumps(m)) for m in messages) // 4
        logger.warning(f"LLM request timed out (~{approx_tokens} estimated tokens)")
        return JSONResponse(status_code=408, content={
            "error": {
                "message": f"Request timed out after {settings.llm_timeout}s. "
                           f"The conversation may be too long (~{approx_tokens} estimated tokens) "
                           f"or the response too large. Please reduce the conversation length.",
                "type": "timeout_error",
                "code": "request_timeout",
            }
        })

    except Exception as e:
        logger.exception("Chat completion error")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(
    rag_chain,
    messages: list,
    request: ChatCompletionRequest,
    use_rag: bool,
) -> AsyncGenerator[str, None]:
    """Generate streaming response."""
    try:
        async for chunk in await rag_chain.query(
            messages=messages,
            collection=request.collection,
            use_rag=use_rag,
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
    settings = get_settings()
    return {
        "object": "list",
        "data": [
            {
                "id": settings.llm_model,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "mindfu",
                "context_window": settings.max_model_len,
            }
        ],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model info (OpenAI-compatible)."""
    settings = get_settings()
    return {
        "id": model_id,
        "object": "model",
        "created": int(datetime.now().timestamp()),
        "owned_by": "mindfu",
        "context_window": settings.max_model_len,
    }
