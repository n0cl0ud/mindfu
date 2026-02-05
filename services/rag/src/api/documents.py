"""
MindFu Documents API - Document Upload and Indexing
"""
import hashlib
import logging
import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams

from ..core.config import get_settings
from ..core.embeddings import get_embedding_service
from ..core.rag_chain import get_rag_chain
from .metrics import record_document_ingested
from ..models.schemas import (
    CollectionCreateRequest,
    CollectionInfo,
    CollectionListResponse,
    DocumentQueryRequest,
    DocumentQueryResponse,
    DocumentQueryResult,
    DocumentUploadRequest,
    DocumentUploadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["documents"])

# Document processors for different file types
SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".html", ".css", ".pdf", ".docx"}


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Get configured text splitter."""
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


async def extract_text_from_file(file: UploadFile) -> str:
    """Extract text content from uploaded file."""
    content = await file.read()
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        from pypdf import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(content))
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        return "\n\n".join(text_parts)

    elif ext == ".docx":
        from docx import Document
        from io import BytesIO

        doc = Document(BytesIO(content))
        return "\n\n".join([para.text for para in doc.paragraphs])

    elif ext == ".html":
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text(separator="\n")

    else:
        # Treat as plain text
        return content.decode("utf-8")


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@router.post("/documents")
async def upload_document(request: DocumentUploadRequest):
    """
    Upload and index a document for RAG.

    The document will be chunked and stored in the vector database.
    Supports deduplication: if a document with the same source URL exists,
    it will be skipped (same content) or updated (different content).
    """
    try:
        settings = get_settings()
        rag_chain = get_rag_chain()
        collection = request.collection or settings.default_collection

        # Compute content hash for deduplication
        content_hash = compute_content_hash(request.content)
        source_url = request.metadata.get("source")

        # Check for existing document with same source
        action = "created"
        if source_url:
            existing = rag_chain.find_by_source(source_url, collection)
            if existing:
                existing_hash = existing[0].get("metadata", {}).get("content_hash")
                if existing_hash == content_hash:
                    # Same content, skip
                    record_document_ingested("skipped")
                    return {
                        "action": "skipped",
                        "reason": "content unchanged",
                        "source": source_url,
                        "collection": collection,
                        "chunks_count": len(existing),
                    }
                else:
                    # Different content, delete old and re-insert
                    deleted = rag_chain.delete_by_source(source_url, collection)
                    logger.info(f"Deleted {deleted} old chunks for {source_url}")
                    action = "updated"

        # Add content_hash to metadata
        metadata = {
            **request.metadata,
            "content_hash": content_hash,
        }

        if request.chunk:
            # Split into chunks
            splitter = get_text_splitter()
            chunks = splitter.split_text(request.content)

            # Add each chunk
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                })

            doc_ids = rag_chain.add_documents(documents, collection)
        else:
            # Add as single document
            doc_id = rag_chain.add_document(
                content=request.content,
                metadata=metadata,
                collection=collection,
            )
            doc_ids = [doc_id]
            chunks = [request.content]

        record_document_ingested(action)
        return {
            "action": action,
            "document_ids": doc_ids,
            "chunks_created": len(chunks),
            "collection": collection,
            "source": source_url,
        }

    except Exception as e:
        logger.exception("Document upload error")
        raise HTTPException(status_code=500, detail=str(e))


def _process_batch_sync(documents: List[dict]) -> dict:
    """
    Synchronous batch processing - runs in thread pool to not block event loop.
    """
    settings = get_settings()
    rag_chain = get_rag_chain()

    results = {
        "created": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "total_chunks": 0,
    }

    # Group by collection
    by_collection = {}
    for doc in documents:
        collection = doc.get("collection") or settings.default_collection
        if collection not in by_collection:
            by_collection[collection] = []
        by_collection[collection].append(doc)

    for collection, docs in by_collection.items():
        to_insert = []
        for doc in docs:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            chunk = doc.get("chunk", True)

            content_hash = compute_content_hash(content)
            source_url = metadata.get("source")

            action = "created"
            if source_url:
                existing = rag_chain.find_by_source(source_url, collection)
                if existing:
                    existing_hash = existing[0].get("metadata", {}).get("content_hash")
                    if existing_hash == content_hash:
                        results["skipped"] += 1
                        continue
                    else:
                        rag_chain.delete_by_source(source_url, collection)
                        action = "updated"

            meta_with_hash = {**metadata, "content_hash": content_hash}

            if chunk:
                splitter = get_text_splitter()
                chunks = splitter.split_text(content)
                for i, chunk_text in enumerate(chunks):
                    to_insert.append({
                        "content": chunk_text,
                        "metadata": {**meta_with_hash, "chunk_index": i, "total_chunks": len(chunks)},
                    })
            else:
                to_insert.append({
                    "content": content,
                    "metadata": meta_with_hash,
                })

            if action == "created":
                results["created"] += 1
            else:
                results["updated"] += 1

        # Batch insert
        if to_insert:
            rag_chain.add_documents(to_insert, collection)
            results["total_chunks"] += len(to_insert)

    return results


@router.post("/documents/batch")
async def upload_documents_batch(documents: List[DocumentUploadRequest]):
    """
    Upload multiple documents in a single batch for faster ingestion.

    - Runs in thread pool to not block other requests
    - Uses batch embedding (much faster than individual)
    - Skips duplicates based on content hash
    """
    import asyncio

    try:
        # Convert Pydantic models to dicts for thread safety
        docs_data = [doc.model_dump() for doc in documents]

        # Run heavy processing in thread pool
        results = await asyncio.to_thread(_process_batch_sync, docs_data)
        return results

    except Exception as e:
        logger.exception("Batch upload error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    collection: Optional[str] = None,
):
    """
    Upload a file for RAG indexing.

    Supported formats: txt, md, py, js, ts, json, yaml, html, pdf, docx
    """
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}",
        )

    try:
        content = await extract_text_from_file(file)

        request = DocumentUploadRequest(
            content=content,
            metadata={"source": filename, "file_type": ext},
            collection=collection,
            chunk=True,
        )

        return await upload_document(request)

    except Exception as e:
        logger.exception("File upload error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/query", response_model=DocumentQueryResponse)
async def query_documents(request: DocumentQueryRequest):
    """
    Query the document store for relevant content.

    Returns the most similar chunks based on semantic similarity.
    """
    try:
        settings = get_settings()
        rag_chain = get_rag_chain()
        collection = request.collection or settings.default_collection

        contexts = rag_chain.retrieve_context(
            query=request.query,
            collection=collection,
            top_k=request.top_k,
        )

        return DocumentQueryResponse(
            results=[
                DocumentQueryResult(
                    content=ctx["content"],
                    metadata=ctx["metadata"],
                    score=ctx["score"],
                )
                for ctx in contexts
            ],
            query=request.query,
            collection=collection,
        )

    except Exception as e:
        logger.exception("Document query error")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Collection Management
# =============================================================================


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    """List all document collections."""
    try:
        rag_chain = get_rag_chain()
        collections = rag_chain.qdrant.get_collections().collections

        return CollectionListResponse(
            collections=[
                CollectionInfo(
                    name=col.name,
                    vectors_count=0,  # Would need additional call to get exact count
                    points_count=0,
                )
                for col in collections
            ]
        )

    except Exception as e:
        logger.exception("List collections error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections")
async def create_collection(request: CollectionCreateRequest):
    """Create a new document collection."""
    try:
        rag_chain = get_rag_chain()
        embedding_service = get_embedding_service()

        vector_size = request.vector_size or embedding_service.dimension

        rag_chain.qdrant.create_collection(
            collection_name=request.name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        return {"message": f"Collection '{request.name}' created successfully"}

    except Exception as e:
        logger.exception("Create collection error")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """Delete a document collection."""
    try:
        rag_chain = get_rag_chain()
        rag_chain.qdrant.delete_collection(collection_name=name)

        return {"message": f"Collection '{name}' deleted successfully"}

    except Exception as e:
        logger.exception("Delete collection error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{name}/stats")
async def get_collection_stats(name: str):
    """Get statistics for a collection."""
    try:
        rag_chain = get_rag_chain()
        info = rag_chain.qdrant.get_collection(collection_name=name)

        return {
            "name": name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
            "config": {
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value,
            },
        }

    except Exception as e:
        logger.exception("Get collection stats error")
        raise HTTPException(status_code=500, detail=str(e))
