"""
Unit and integration tests for Root and Health endpoints.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Verify that root endpoint responds with application metadata."""
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AgentX"
    assert "version" in data
    assert "health_check" in data


@pytest.mark.asyncio
async def test_health_check_endpoint(async_client: AsyncClient):
    """Verify health check endpoint returns 200 and valid schema."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "AgentX"
    assert "version" in data
    assert "timestamp" in data
    assert "services" in data
    assert "api" in data["services"]
    assert data["services"]["api"]["status"] == "healthy"


@pytest.mark.asyncio
async def test_process_time_header(async_client: AsyncClient):
    """Verify that the middleware injects process time header."""
    response = await async_client.get("/api/v1/health")
    assert "x-process-time-ms" in response.headers
