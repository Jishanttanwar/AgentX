"""
LangGraph Checkpointer & Short-Term Session Memory.
Provides state checkpointing, rewind points, and session persistence.
"""

from typing import Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app.core.logging import get_logger

logger = get_logger("memory.checkpointer")

# Global in-memory checkpointer singleton for active sessions
_memory_saver_instance: Optional[MemorySaver] = None


def get_checkpointer() -> BaseCheckpointSaver:
    """
    Returns the configured LangGraph checkpointer instance.
    Uses MemorySaver for short-term session state (development & test),
    designed for zero-friction migration to PostgresSaver in Phase 7.
    """
    global _memory_saver_instance
    if _memory_saver_instance is None:
        logger.info("Initializing LangGraph MemorySaver for short-term state persistence")
        _memory_saver_instance = MemorySaver()
    return _memory_saver_instance


def reset_checkpointer() -> None:
    """Reset the checkpointer instance (primarily for isolated test fixtures)."""
    global _memory_saver_instance
    _memory_saver_instance = None

