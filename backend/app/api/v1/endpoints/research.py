"""
Research API endpoints for workflow initialization, planning, and task management.
"""

import time
from fastapi import APIRouter, HTTPException, status
from app.agents.planner import planner_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.plan import (
    PlanGenerationRequest,
    PlanGenerationResponse,
)

router = APIRouter()
logger = get_logger("api.research")


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
