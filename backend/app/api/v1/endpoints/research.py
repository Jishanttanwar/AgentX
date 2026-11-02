"""
Research API endpoints for planning, execution, and stateful LangGraph workflow management.
"""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.planner import planner_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.events import WorkflowStateSummary, WorkflowStatus
from app.schemas.intelligence import (
    AdditionalResearchRequest,
    AnalyticalInsight,
    VerificationResult,
    VerificationSummary,
)
from app.schemas.plan import (
    PlanGenerationRequest,
    PlanGenerationResponse,
    ResearchDepth,
)
from app.schemas.research import MultiAgentResearchResult
from app.services.research_coordinator import research_coordinator
from app.services.workflow_service import workflow_service

router = APIRouter()
logger = get_logger("api.research")


class ResearchExecutionRequest(BaseModel):
    """Payload to trigger end-to-end multi-agent research across Planner and Researchers."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="User's research query",
        examples=["Research the future of artificial intelligence in healthcare."],
    )
    depth: ResearchDepth = Field(
        default=ResearchDepth.STANDARD,
        description="Depth level of research: quick, standard, or comprehensive",
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Optional supplementary background or domain constraints",
    )


class ResearchExecutionResponse(BaseModel):
    """Response containing the aggregated plan, sources, and verified findings."""

    success: bool = True
    result: MultiAgentResearchResult
    message: str = "Research execution completed successfully"


class StartWorkflowRequest(BaseModel):
    """Payload to start a stateful LangGraph research session."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Research topic or question",
    )
    depth: ResearchDepth = Field(
        default=ResearchDepth.STANDARD,
        description="Depth level: quick, standard, or comprehensive",
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Optional domain constraints or guidelines",
    )
    human_approval_required: bool = Field(
        default=False,
        description="Whether to halt workflow after planning to await user approval",
    )


class WorkflowActionResponse(BaseModel):
    """Response returned upon workflow initiation or resumption."""

    success: bool = True
    research_id: str
    status: WorkflowStatus
    message: str


class IntelligenceSummaryResponse(BaseModel):
    """Intelligence phase output: analytical insights and fact verification results."""

    research_id: str
    insights: List[AnalyticalInsight] = Field(default_factory=list)
    insights_count: int = 0
    verification_results: List[VerificationResult] = Field(default_factory=list)
    verification_summary: Optional[VerificationSummary] = None
    additional_requests: List[AdditionalResearchRequest] = Field(default_factory=list)
    research_cycle_count: int = 0


# ---------------------------------------------------------------------------
# Basic Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/plan",
    response_model=PlanGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Research Plan",
    description="Invokes the Planner Agent to analyze the user's research query and produce a structured, multi-task execution plan.",
)
async def generate_research_plan(
    request: PlanGenerationRequest,
) -> PlanGenerationResponse:
    """Generate a structured research plan from a query."""
    start_time = time.time()
    try:
        plan = await planner_agent.generate_plan(
            query=request.query,
            depth=request.depth,
            context=request.additional_context,
        )
        duration_ms = (time.time() - start_time) * 1000

        return PlanGenerationResponse(
            success=True,
            plan=plan,
            model_used=settings.OPENAI_MODEL,
            execution_time_ms=round(duration_ms, 2),
        )

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error(f"Error handling research plan request: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate research plan: {str(exc)}",
        )


@router.post(
    "/execute",
    response_model=ResearchExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Multi-Agent Research",
    description="Orchestrates Planner and parallel Researcher agents to gather sources, extract evidence, and organize findings.",
)
async def execute_multi_agent_research(
    request: ResearchExecutionRequest,
) -> ResearchExecutionResponse:
    """Run collaborative Planner -> Parallel Researcher workflow."""
    try:
        result = await research_coordinator.run_research_pipeline(
            query=request.query,
            depth=request.depth,
            context=request.additional_context,
        )
        return ResearchExecutionResponse(
            success=True,
            result=result,
            message=(
                f"Multi-agent research completed with {len(result.all_sources)} sources "
                f"and {len(result.all_findings)} verified findings."
            ),
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error(f"Multi-agent research execution error: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-agent research failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# LangGraph Stateful Workflow Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/workflow/start",
    response_model=WorkflowActionResponse,
    status_code=status.HTTP_200_OK,
    summary="Start Stateful LangGraph Workflow",
    description="Initializes a LangGraph workflow with short-term checkpointing and optional human approval gating.",
)
async def start_research_workflow(
    request: StartWorkflowRequest,
) -> WorkflowActionResponse:
    """Initiate a stateful LangGraph research session."""
    try:
        final_state = await workflow_service.start_workflow(
            query=request.query,
            depth=request.depth,
            context=request.additional_context,
            human_approval_required=request.human_approval_required,
        )
        rid = final_state.get("research_id", "unknown")
        current_status = final_state.get("execution_status", WorkflowStatus.INITIALIZING)

        msg = (
            "Workflow paused awaiting human plan approval"
            if current_status == WorkflowStatus.AWAITING_APPROVAL
            else "Workflow executed successfully through LangGraph"
        )

        return WorkflowActionResponse(
            success=True,
            research_id=rid,
            status=current_status,
            message=msg,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start workflow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow error: {str(e)}",
        )


@router.get(
    "/workflow/{research_id}/summary",
    response_model=WorkflowStateSummary,
    status_code=status.HTTP_200_OK,
    summary="Get Workflow Summary",
    description="Retrieves high-level metrics, progress, and status for a workflow session.",
)
async def get_workflow_summary(research_id: str) -> WorkflowStateSummary:
    """Retrieve summary of workflow state."""
    summary = await workflow_service.get_summary(research_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research workflow session '{research_id}' not found.",
        )
    return summary


@router.get(
    "/workflow/{research_id}/state",
    status_code=status.HTTP_200_OK,
    summary="Get Full Workflow State",
    description="Retrieves the full checkpointed state of the multi-agent graph.",
)
async def get_workflow_state(research_id: str) -> Dict[str, Any]:
    """Retrieve full raw state for a workflow session."""
    state = await workflow_service.get_state(research_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research workflow session '{research_id}' not found.",
        )
    return dict(state)


@router.post(
    "/workflow/{research_id}/approve",
    response_model=WorkflowActionResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve Research Plan",
    description="Approves a pending research plan and resumes the LangGraph execution.",
)
async def approve_research_workflow(research_id: str) -> WorkflowActionResponse:
    """Approve a paused workflow and resume execution."""
    try:
        final_state = await workflow_service.approve_and_resume(research_id)
        current_status = final_state.get("execution_status", WorkflowStatus.COMPLETED)
        return WorkflowActionResponse(
            success=True,
            research_id=research_id,
            status=current_status,
            message="Plan approved and workflow resumed to completion.",
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to approve workflow {research_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Approval error: {str(e)}",
        )


@router.get(
    "/workflow/{research_id}/intelligence",
    response_model=IntelligenceSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Intelligence Phase Results",
    description=(
        "Returns the Analyst's synthesized insights and the Fact Checker's verification "
        "results for a completed or in-progress intelligence phase."
    ),
)
async def get_intelligence_summary(research_id: str) -> IntelligenceSummaryResponse:
    """Retrieve analytical insights and fact verification results for a workflow."""
    state = await workflow_service.get_state(research_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research workflow session '{research_id}' not found.",
        )

    insights = state.get("analytical_insights", [])
    verification_results = state.get("verification_results", [])
    additional_requests = state.get("additional_research_requests", [])
    research_cycle_count = state.get("research_cycle_count", 0)

    # Build verification summary if results exist
    verification_summary: Optional[VerificationSummary] = None
    if verification_results:
        from app.schemas.intelligence import VerificationStatus
        verified = sum(1 for v in verification_results if v.status == VerificationStatus.VERIFIED)
        partial = sum(1 for v in verification_results if v.status == VerificationStatus.PARTIALLY_SUPPORTED)
        conflict = sum(1 for v in verification_results if v.status == VerificationStatus.CONFLICTING_EVIDENCE)
        unsupported = sum(1 for v in verification_results if v.status == VerificationStatus.UNSUPPORTED)
        total = len(verification_results)
        verification_summary = VerificationSummary(
            total_claims_checked=total,
            verified_count=verified,
            partially_supported_count=partial,
            conflicting_count=conflict,
            unsupported_count=unsupported,
            verification_rate=round((verified + partial) / total, 3) if total > 0 else 0.0,
            needs_additional_research=len(additional_requests) > 0,
            additional_requests=additional_requests,
        )

    return IntelligenceSummaryResponse(
        research_id=research_id,
        insights=insights,
        insights_count=len(insights),
        verification_results=verification_results,
        verification_summary=verification_summary,
        additional_requests=additional_requests,
        research_cycle_count=research_cycle_count,
    )
