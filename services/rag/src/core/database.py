"""
MindFu Database - PostgreSQL Models and Connection
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, List, Optional
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy base class."""
    pass


class Conversation(Base):
    """Conversation log table."""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Request data
    model = Column(String(255), nullable=False)
    messages = Column(JSON, nullable=False)  # List of messages

    # Response data
    response_content = Column(Text, nullable=True)
    response_tool_calls = Column(JSON, nullable=True)
    finish_reason = Column(String(50), nullable=True)

    # Metadata
    tokens_prompt = Column(Integer, nullable=True)
    tokens_completion = Column(Integer, nullable=True)
    tokens_total = Column(Integer, nullable=True)

    # RAG context used
    rag_contexts_used = Column(Integer, default=0)
    rag_sources = Column(JSON, nullable=True)

    # For training export
    exported_at = Column(DateTime, nullable=True)
    training_quality = Column(String(20), nullable=True)  # good, bad, skip


# Async engine and session
_async_engine = None
_async_session_factory = None


def get_async_engine():
    """Get async SQLAlchemy engine."""
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.postgres_url,
            echo=False,
            pool_pre_ping=True,
        )
    return _async_engine


def get_async_session_factory():
    """Get async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_async_engine()
        _async_session_factory = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_database():
    """Initialize database tables."""
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


class ConversationLogger:
    """Service for logging conversations."""

    async def log_conversation(
        self,
        model: str,
        messages: List[dict],
        response_content: Optional[str] = None,
        response_tool_calls: Optional[List[dict]] = None,
        finish_reason: Optional[str] = None,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        tokens_total: Optional[int] = None,
        rag_contexts_used: int = 0,
        rag_sources: Optional[List[str]] = None,
    ) -> str:
        """Log a conversation to the database."""
        async with get_session() as session:
            conversation = Conversation(
                model=model,
                messages=messages,
                response_content=response_content,
                response_tool_calls=response_tool_calls,
                finish_reason=finish_reason,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                tokens_total=tokens_total,
                rag_contexts_used=rag_contexts_used,
                rag_sources=rag_sources,
            )
            session.add(conversation)
            await session.flush()

            logger.debug(f"Logged conversation {conversation.id}")
            return conversation.id

    async def get_conversations(
        self,
        limit: int = 100,
        offset: int = 0,
        model: Optional[str] = None,
        unexported_only: bool = False,
    ) -> List[Conversation]:
        """Get conversations from the database."""
        from sqlalchemy import select

        async with get_session() as session:
            query = select(Conversation).order_by(Conversation.created_at.desc())

            if model:
                query = query.where(Conversation.model == model)
            if unexported_only:
                query = query.where(Conversation.exported_at.is_(None))

            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def mark_exported(self, conversation_ids: List[str]) -> int:
        """Mark conversations as exported for training."""
        from sqlalchemy import update

        async with get_session() as session:
            stmt = (
                update(Conversation)
                .where(Conversation.id.in_(conversation_ids))
                .values(exported_at=datetime.utcnow())
            )
            result = await session.execute(stmt)
            return result.rowcount

    async def set_quality(self, conversation_id: str, quality: str) -> bool:
        """Set training quality rating for a conversation."""
        from sqlalchemy import update

        if quality not in ("good", "bad", "skip"):
            raise ValueError("Quality must be 'good', 'bad', or 'skip'")

        async with get_session() as session:
            stmt = (
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(training_quality=quality)
            )
            result = await session.execute(stmt)
            return result.rowcount > 0

    async def export_for_training(
        self,
        quality_filter: Optional[str] = "good",
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Export conversations in training format."""
        from sqlalchemy import select

        async with get_session() as session:
            query = select(Conversation).where(
                Conversation.response_content.isnot(None)
            )

            if quality_filter:
                query = query.where(Conversation.training_quality == quality_filter)

            query = query.order_by(Conversation.created_at.desc())

            if limit:
                query = query.limit(limit)

            result = await session.execute(query)
            conversations = list(result.scalars().all())

            # Format for training (Alpaca-style)
            training_data = []
            for conv in conversations:
                # Get the last user message as instruction
                user_messages = [m for m in conv.messages if m.get("role") == "user"]
                if not user_messages:
                    continue

                instruction = user_messages[-1].get("content", "")

                # Get system prompt if any
                system_messages = [m for m in conv.messages if m.get("role") == "system"]
                system_prompt = system_messages[0].get("content", "") if system_messages else ""

                training_data.append({
                    "instruction": instruction,
                    "input": system_prompt,
                    "output": conv.response_content,
                    "metadata": {
                        "conversation_id": conv.id,
                        "model": conv.model,
                        "created_at": conv.created_at.isoformat(),
                    }
                })

            return training_data

    async def get_stats(self) -> dict:
        """Get conversation statistics."""
        from sqlalchemy import func, select

        async with get_session() as session:
            # Total count
            total = await session.scalar(select(func.count(Conversation.id)))

            # Count by quality
            quality_counts = {}
            for quality in ("good", "bad", "skip", None):
                if quality is None:
                    count = await session.scalar(
                        select(func.count(Conversation.id))
                        .where(Conversation.training_quality.is_(None))
                    )
                    quality_counts["unrated"] = count
                else:
                    count = await session.scalar(
                        select(func.count(Conversation.id))
                        .where(Conversation.training_quality == quality)
                    )
                    quality_counts[quality] = count

            # Exported count
            exported = await session.scalar(
                select(func.count(Conversation.id))
                .where(Conversation.exported_at.isnot(None))
            )

            # Token totals
            total_tokens = await session.scalar(
                select(func.sum(Conversation.tokens_total))
            ) or 0

            return {
                "total_conversations": total,
                "by_quality": quality_counts,
                "exported": exported,
                "total_tokens": total_tokens,
            }


# Singleton instance
_conversation_logger: Optional[ConversationLogger] = None


def get_conversation_logger() -> ConversationLogger:
    """Get conversation logger instance."""
    global _conversation_logger
    if _conversation_logger is None:
        _conversation_logger = ConversationLogger()
    return _conversation_logger
