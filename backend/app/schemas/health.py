"""
Health check schema models.
"""

from datetime import datetime, timezone
from typing import Dict, Optional
from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Status details for internal or external dependencies."""

    status: str = Field(description="Service status (healthy, degraded, unavailable)")
    latency_ms: Optional[float] = Field(default=None, description="Check latency in ms")
    details: Optional[str] = Field(default=None, description="Optional extra detail")


class HealthResponse(BaseModel):
    """System health check payload."""

    status: str = Field(default="healthy", description="Overall health status")
    app_name: str = Field(description="Name of the application")
    version: str = Field(description="Semantic version of application")
    environment: str = Field(description="Current deployment environment")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Current UTC timestamp",
    )
    services: Dict[str, ServiceStatus] = Field(
        default_factory=dict,
        description="Status of connected services (db, redis, llm)",
    )
