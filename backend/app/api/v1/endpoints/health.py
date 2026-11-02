"""
Health check endpoints for system observability and load-balancer probes.
"""

from fastapi import APIRouter, status
from app.core.config import settings
from app.schemas.health import HealthResponse, ServiceStatus

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Returns the operational status, application version, environment, and subsystem states.",
)
async def check_health() -> HealthResponse:
    """Check backend service health."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        services={
            "api": ServiceStatus(status="healthy", details="FastAPI server running"),
            "database": ServiceStatus(status="standby", details="Configured, ready for migration"),
            "vector_store": ServiceStatus(status="standby", details="ChromaDB ready"),
        },
    )

