"""
AgentState Definition for AgentX LangGraph Multi-Agent Architecture.
Uses TypedDict with Annotated reducers to ensure concurrency-safe state transitions.
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict
import uuid

from app.graph.reducers import (
    reduce_findings,
    reduce_findings_by_task,
    reduce_logs,
    reduce_sources,
    reduce_token_usage,
)
from app.schemas.events import ExecutionLogEvent, WorkflowStatus
from app.schemas.plan import ResearchDepth, ResearchPlan, ResearchTask
from app.schemas.research import ResearchFinding, SearchResult, SourceMetadata


class AgentState(TypedDict, total=False):
    """
    Centralized LangGraph state schema.
    Shared across all seven specialized agents with conflict-free reducers.
    """

    # Workflow Identity & Input
    research_id: str
    user_query: str
    research_depth: ResearchDepth
    additional_context: Optional[str]

    # Planning Phase State
    research_plan: Optional[ResearchPlan]
    research_tasks: List[ResearchTask]
    assigned_agents: List[str]

    # Human-in-the-Loop Governance
    human_approval_required: bool
    human_approved: bool

    # Research Phase State (Parallel Accumulation)
    collected_sources: Annotated[List[SourceMetadata], reduce_sources]
    research_findings: Annotated[List[ResearchFinding], reduce_findings]
    findings_by_task: Annotated[Dict[str, List[ResearchFinding]], reduce_findings_by_task]
    search_queries_executed: Annotated[List[str], lambda x, y: list(set((x or []) + (y or [])))]

    # Analysis & Verification State (Phases 5)
    analytical_insights: List[Dict[str, Any]]
    extracted_claims: List[Dict[str, Any]]
    verification_results: List[Dict[str, Any]]
    additional_research_requests: List[str]

    # Report Synthesis & Review State (Phase 6)
    generated_report: Optional[Dict[str, Any]]
    reviewer_feedback: List[str]
    quality_score: Optional[float]
    revision_count: int

    # Observability, Telemetry & Diagnostics
    execution_status: WorkflowStatus
    current_step: str
    execution_logs: Annotated[List[ExecutionLogEvent], reduce_logs]
    token_usage: Annotated[Dict[str, int], reduce_token_usage]
    error_details: Optional[str]


def create_initial_state(
    query: str,
    research_id: Optional[str] = None,
    depth: ResearchDepth = ResearchDepth.STANDARD,
    context: Optional[str] = None,
    human_approval_required: bool = False,
) -> AgentState:
    """
    Create a clean, standardized initial AgentState dictionary for a new workflow.
    """
    rid = research_id or f"res_{uuid.uuid4().hex[:12]}"

    init_event = ExecutionLogEvent(
        research_id=rid,
        agent_name="orchestrator",
        event_type="transition",
        message=f"AgentX research workflow initialized for query='{query[:80]}'",
        details={"depth": depth.value, "approval_required": human_approval_required},
    )

    return {
        "research_id": rid,
        "user_query": query.strip(),
        "research_depth": depth,
        "additional_context": context,
        "research_plan": None,
        "research_tasks": [],
        "assigned_agents": ["orchestrator", "planner"],
        "human_approval_required": human_approval_required,
        "human_approved": False,
        "collected_sources": [],
        "research_findings": [],
        "findings_by_task": {},
        "search_queries_executed": [],
        "analytical_insights": [],
        "extracted_claims": [],
        "verification_results": [],
        "additional_research_requests": [],
        "generated_report": None,
        "reviewer_feedback": [],
        "quality_score": None,
        "revision_count": 0,
        "execution_status": WorkflowStatus.INITIALIZING,
        "current_step": "orchestrator_init",
        "execution_logs": [init_event],
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "error_details": None,
    }
