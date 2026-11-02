"""
Workflow event, status, and audit log schemas for LangGraph state machine.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    """Lifecycle statuses for the LangGraph state machine."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    VERIFYING = "verifying"
    WRITING = "writing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionLogEvent(BaseModel):
    """Individual structured audit log event emitted during graph execution."""

    event_id: str = Field(
        default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}",
        description="Unique event identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of event generation in UTC",
    )
    research_id: str = Field(..., description="Active research workflow session ID")
    agent_name: str = Field(..., description="Name of the agent generating the log")
    event_type: str = Field(
        default="info",
        description="Category of event: info, warning, error, transition, artifact",
    )
    message: str = Field(..., description="Human-readable event message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured context or metadata",
    )


class WorkflowStateSummary(BaseModel):
    """High-level summary of a running or completed research workflow."""

    research_id: str
    status: WorkflowStatus
    query: str
    current_step: str
    tasks_total: int = 0
    tasks_completed: int = 0
    sources_count: int = 0
    findings_count: int = 0
    human_approval_required: bool = False
    error_details: Optional[str] = None
    execution_time_ms: float = 0.0
