"""
Research API endpoints for workflow initialization, planning, and multi-agent execution.
"""

import time
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.planner import planner_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.plan import (
    PlanGenerationRequest,
    PlanGenerationResponse,
    ResearchDepth,
)
from app.schemas.research import MultiAgentResearchResult
from app.services.research_coordinator import research_coordinator

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
