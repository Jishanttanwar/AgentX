"""
AgentState Definition for AgentX LangGraph Multi-Agent Architecture.
Uses TypedDict with Annotated reducers to ensure concurrency-safe state transitions.
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict
import uuid

from app.graph.reducers import (
    reduce_additional_requests,
    reduce_claims,
    reduce_findings,
    reduce_findings_by_task,
    reduce_insights,
    reduce_logs,
    reduce_sources,
    reduce_token_usage,
    reduce_verification_results,
)
from app.schemas.events import ExecutionLogEvent, WorkflowStatus
from app.schemas.intelligence import (
    AdditionalResearchRequest,
    AnalyticalInsight,
    FactualClaim,
    VerificationResult,
)
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
    research_cycle_count: int
    max_research_cycles: int

    # Analysis & Verification State (Phase 5)
    analytical_insights: Annotated[List[AnalyticalInsight], reduce_insights]
    extracted_claims: Annotated[List[FactualClaim], reduce_claims]
    verification_results: Annotated[List[VerificationResult], reduce_verification_results]
    additional_research_requests: Annotated[List[AdditionalResearchRequest], reduce_additional_requests]

    # Report Synthesis & Review State (Phase 6)
    generated_report: Optional[Dict[str, Any]]
    reviewer_feedback: List[str]
    quality_score: Optional[float]
    revision_count: int
    max_revision_cycles: int

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
    max_research_cycles: int = 2,
    max_revision_cycles: int = 3,
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
        "research_cycle_count": 0,
        "max_research_cycles": max_research_cycles,
        "analytical_insights": [],
        "extracted_claims": [],
        "verification_results": [],
        "additional_research_requests": [],
        "generated_report": None,
        "reviewer_feedback": [],
        "quality_score": None,
        "revision_count": 0,
        "max_revision_cycles": max_revision_cycles,
        "execution_status": WorkflowStatus.INITIALIZING,
        "current_step": "orchestrator_init",
        "execution_logs": [init_event],
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "error_details": None,
    }
