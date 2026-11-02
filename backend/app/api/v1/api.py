"""
API v1 Router aggregation.
Mounts individual feature endpoints into the /api/v1 path namespace.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import health, research

api_router = APIRouter()

# Register endpoint routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(research.router, prefix="/research", tags=["Research"])
