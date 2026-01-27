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
            request_data = {
                "model": kwargs.get("model", self.settings.llm_model),
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

            response = await client.post(
                f"{self.settings.llm_base_url}/chat/completions",
                json=request_data,
            )
            response.raise_for_status()
            result = response.json()

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
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            # Build request data
            request_data = {
                "model": kwargs.get("model", self.settings.llm_model),
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "stream": True,
            }

            # Forward tool-related parameters
            if kwargs.get("tools"):
                request_data["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice") is not None:
                request_data["tool_choice"] = kwargs["tool_choice"]

            # Add other kwargs
            handled_keys = ["model", "temperature", "max_tokens", "stream", "tools", "tool_choice"]
            for k, v in kwargs.items():
                if k not in handled_keys and v is not None:
                    request_data[k] = v

            async with client.stream(
                "POST",
                f"{self.settings.llm_base_url}/chat/completions",
                json=request_data,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line + "\n\n"

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
