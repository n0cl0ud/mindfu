"""
MindFu RAG Chain Implementation
"""
import logging
from typing import AsyncGenerator, List, Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import get_settings
from .embeddings import get_embedding_service

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG chain for context-augmented LLM queries."""

    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.qdrant = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the default collection exists."""
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.settings.default_collection not in collection_names:
            self.qdrant.create_collection(
                collection_name=self.settings.default_collection,
                vectors_config=VectorParams(
                    size=self.embedding_service.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.settings.default_collection}")

    def retrieve_context(
        self,
        query: str,
        collection: str | None = None,
        top_k: int | None = None,
    ) -> List[dict]:
        """Retrieve relevant context from vector store."""
        collection = collection or self.settings.default_collection
        top_k = top_k or self.settings.top_k

        query_embedding = self.embedding_service.embed_text(query)

        results = self.qdrant.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
        )

        contexts = []
        for result in results:
            contexts.append({
                "content": result.payload.get("content", ""),
                "metadata": result.payload.get("metadata", {}),
                "score": result.score,
            })

        return contexts

    def format_context(self, contexts: List[dict]) -> str:
        """Format retrieved contexts into a string."""
        if not contexts:
            return ""

        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get("metadata", {}).get("source", "Unknown")
            content = ctx.get("content", "")
            context_parts.append(f"[Source {i}: {source}]\n{content}")

        return "\n\n".join(context_parts)

    def augment_messages(
        self,
        messages: List[dict],
        contexts: List[dict],
    ) -> List[dict]:
        """Augment messages with retrieved context."""
        if not contexts:
            return messages

        context_text = self.format_context(contexts)
        system_augmentation = f"""You have access to the following relevant context from the knowledge base:

<context>
{context_text}
</context>

Use this context to inform your response when relevant. If the context doesn't contain relevant information, respond based on your general knowledge.

IMPORTANT: When multiple versions of documentation are present in the context, always prefer information from the most recent version unless the user specifically asks about an older version."""

        # Find or create system message
        augmented = []
        has_system = False

        for msg in messages:
            if msg.get("role") == "system":
                has_system = True
                augmented.append({
                    "role": "system",
                    "content": f"{msg['content']}\n\n{system_augmentation}",
                })
            else:
                augmented.append(msg)

        if not has_system:
            augmented.insert(0, {
                "role": "system",
                "content": system_augmentation,
            })

        return augmented

    async def query(
        self,
        messages: List[dict],
        collection: str | None = None,
        use_rag: bool = True,
        stream: bool = False,
        **kwargs,
    ) -> dict | AsyncGenerator:
        """Execute RAG-augmented query."""
        # Get the last user message for context retrieval
        user_query = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # Retrieve and augment context if RAG is enabled
        contexts = []
        if use_rag and user_query:
            contexts = self.retrieve_context(user_query, collection)
            messages = self.augment_messages(messages, contexts)

        # Call LLM
        if stream:
            return self._stream_completion(messages, contexts, **kwargs)
        else:
            return await self._completion(messages, contexts, **kwargs)

    async def _completion(
        self,
        messages: List[dict],
        contexts: List[dict],
        **kwargs,
    ) -> dict:
        """Execute non-streaming completion."""
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            # Build request, excluding None values for llama.cpp compatibility
            # Always use configured model - clients may send short names that vLLM doesn't recognize
            request_data = {
                "model": self.settings.llm_model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 2048),
            }

            # Forward tool-related parameters
            if kwargs.get("tools"):
                request_data["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice") is not None:
                request_data["tool_choice"] = kwargs["tool_choice"]

            # Add optional kwargs, excluding None values and already handled keys
            handled_keys = ["model", "temperature", "max_tokens", "tools", "tool_choice"]
            for k, v in kwargs.items():
                if k not in handled_keys and v is not None:
                    request_data[k] = v

            # DEBUG: Log request data for tool calls
            if kwargs.get("tools"):
                import json
                logger.info(f"DEBUG Tool call request to vLLM:")
                logger.info(f"  Model: {request_data.get('model')}")
                logger.info(f"  Messages count: {len(request_data.get('messages', []))}")
                logger.info(f"  Tools count: {len(request_data.get('tools', []))}")
                logger.info(f"  Full request: {json.dumps(request_data, default=str)[:2000]}")

            response = await client.post(
                f"{self.settings.llm_base_url}/chat/completions",
                json=request_data,
            )
            response.raise_for_status()
            result = response.json()

            # DEBUG: Log response for tool calls
            if kwargs.get("tools") and result.get("choices"):
                choice = result["choices"][0]
                msg = choice.get("message", {})
                if msg.get("tool_calls"):
                    import json
                    logger.info(f"DEBUG Tool call response from vLLM:")
                    logger.info(f"  Tool calls: {json.dumps(msg.get('tool_calls'), default=str)[:1000]}")

            # Add context metadata
            result["_rag_context"] = {
                "contexts_used": len(contexts),
                "sources": [c.get("metadata", {}).get("source") for c in contexts],
            }

            return result

    async def _stream_completion(
        self,
        messages: List[dict],
        contexts: List[dict],
        **kwargs,
    ) -> AsyncGenerator:
        """Execute streaming completion."""
        import json

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            # Build request data
            # Always include usage in streaming responses for client compatibility
            stream_options = kwargs.get("stream_options") or {}
            stream_options["include_usage"] = True

            # Always use configured model - clients may send short names that vLLM doesn't recognize
            request_data = {
                "model": self.settings.llm_model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "stream": True,
                "stream_options": stream_options,
            }

            # Forward tool-related parameters
            has_tools = bool(kwargs.get("tools"))
            if has_tools:
                request_data["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice") is not None:
                request_data["tool_choice"] = kwargs["tool_choice"]

            # Add other kwargs
            handled_keys = ["model", "temperature", "max_tokens", "stream", "stream_options", "tools", "tool_choice"]
            for k, v in kwargs.items():
                if k not in handled_keys and v is not None:
                    request_data[k] = v

            async with client.stream(
                "POST",
                f"{self.settings.llm_base_url}/chat/completions",
                json=request_data,
            ) as response:
                response.raise_for_status()

                if not has_tools:
                    # No tools - stream directly
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            yield line + "\n\n"
                else:
                    # Buffer chunks and reconstruct tool_calls for Vibe compatibility
                    # Vibe can't accumulate incremental tool_call arguments
                    chunks = []
                    tool_calls = {}  # index -> {id, type, function: {name, arguments}}
                    role = None
                    content_parts = []
                    finish_reason = None
                    usage = None
                    chunk_id = None
                    created = None
                    model = None

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(data)
                            chunk_id = chunk.get("id", chunk_id)
                            created = chunk.get("created", created)
                            model = chunk.get("model", model)

                            if chunk.get("usage"):
                                usage = chunk["usage"]

                            choices = chunk.get("choices", [])
                            if choices:
                                choice = choices[0]
                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]

                                delta = choice.get("delta", {})
                                if delta.get("role"):
                                    role = delta["role"]
                                if delta.get("content"):
                                    content_parts.append(delta["content"])

                                # Accumulate tool_calls
                                for tc in delta.get("tool_calls", []):
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls:
                                        tool_calls[idx] = {
                                            "id": tc.get("id", ""),
                                            "type": tc.get("type", "function"),
                                            "function": {"name": "", "arguments": ""}
                                        }
                                    if tc.get("id"):
                                        tool_calls[idx]["id"] = tc["id"]
                                    if tc.get("type"):
                                        tool_calls[idx]["type"] = tc["type"]
                                    func = tc.get("function", {})
                                    if func.get("name"):
                                        tool_calls[idx]["function"]["name"] = func["name"]
                                    if func.get("arguments"):
                                        tool_calls[idx]["function"]["arguments"] += func["arguments"]
                        except json.JSONDecodeError:
                            continue

                    # Now emit reconstructed chunks
                    # Chunk 1: role + content + complete tool_calls
                    first_delta = {"role": role or "assistant"}
                    if content_parts:
                        first_delta["content"] = "".join(content_parts)
                    if tool_calls:
                        # Convert dict to sorted list
                        tc_list = [tool_calls[i] for i in sorted(tool_calls.keys())]
                        first_delta["tool_calls"] = [
                            {"index": i, **tc} for i, tc in enumerate(tc_list)
                        ]
                        logger.info(f"Reconstructed tool_calls: {json.dumps(tc_list)[:500]}")

                    first_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": first_delta, "logprobs": None, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(first_chunk)}\n\n"

                    # Final chunk: finish_reason + usage
                    final_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish_reason or "stop"}]
                    }
                    if usage:
                        final_chunk["usage"] = usage
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

    def add_document(
        self,
        content: str,
        metadata: dict | None = None,
        collection: str | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Add a document to the vector store."""
        import uuid

        collection = collection or self.settings.default_collection
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}

        embedding = self.embedding_service.embed_text(content)

        self.qdrant.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "metadata": metadata,
                    },
                )
            ],
        )

        return doc_id

    def find_by_source(
        self,
        source_url: str,
        collection: str | None = None,
    ) -> List[dict]:
        """Find documents by source URL in metadata."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        collection = collection or self.settings.default_collection

        results = self.qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source_url),
                    )
                ]
            ),
            limit=1000,  # Max chunks per document
            with_payload=True,
            with_vectors=False,
        )

        points = results[0] if results else []
        return [
            {
                "id": p.id,
                "content": p.payload.get("content", ""),
                "metadata": p.payload.get("metadata", {}),
            }
            for p in points
        ]

    def delete_by_source(
        self,
        source_url: str,
        collection: str | None = None,
    ) -> int:
        """Delete all documents with given source URL."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        collection = collection or self.settings.default_collection

        # Get count before delete
        existing = self.find_by_source(source_url, collection)
        count = len(existing)

        if count > 0:
            self.qdrant.delete(
                collection_name=collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source_url),
                        )
                    ]
                ),
            )

        return count

    def add_documents(
        self,
        documents: List[dict],
        collection: str | None = None,
    ) -> List[str]:
        """Add multiple documents to the vector store."""
        import uuid

        collection = collection or self.settings.default_collection

        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_service.embed_texts(contents)

        points = []
        doc_ids = []

        for doc, embedding in zip(documents, embeddings):
            doc_id = doc.get("id") or str(uuid.uuid4())
            doc_ids.append(doc_id)

            points.append(
                PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                    },
                )
            )

        self.qdrant.upsert(
            collection_name=collection,
            points=points,
        )

        return doc_ids


_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Get RAG chain instance."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
