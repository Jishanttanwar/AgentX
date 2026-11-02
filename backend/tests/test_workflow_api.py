"""
Integration tests for LangGraph stateful workflow REST API endpoints.
"""

from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient

from app.schemas.events import WorkflowStateSummary, WorkflowStatus
from app.schemas.plan import ResearchDepth


@pytest.mark.asyncio
async def test_api_start_workflow_success(async_client: AsyncClient):
    """Verify POST /api/v1/research/workflow/start initiates a stateful session."""
    mock_final_state = {
        "research_id": "res_test_123",
        "execution_status": WorkflowStatus.COMPLETED,
        "user_query": "AI in Drug Discovery",
    }

    with patch(
        "app.api.v1.endpoints.research.workflow_service.start_workflow",
        new_callable=AsyncMock,
    ) as mock_start:
        mock_start.return_value = mock_final_state

        payload = {
            "query": "AI in Drug Discovery",
            "depth": "standard",
            "human_approval_required": False,
        }
        response = await async_client.post("/api/v1/research/workflow/start", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["research_id"] == "res_test_123"
        assert data["status"] == "completed"


@pytest.mark.asyncio
async def test_api_workflow_summary_found_and_not_found(async_client: AsyncClient):
    """Verify GET /api/v1/research/workflow/{id}/summary behavior."""
    mock_summary = WorkflowStateSummary(
        research_id="res_test_123",
        status=WorkflowStatus.RESEARCHING,
        query="AI in Drug Discovery",
        current_step="researcher_worker",
        tasks_total=3,
        tasks_completed=1,
        sources_count=4,
        findings_count=6,
        human_approval_required=False,
    )

    with patch(
        "app.api.v1.endpoints.research.workflow_service.get_summary",
        new_callable=AsyncMock,
    ) as mock_get:
        # 1. Found case
        mock_get.return_value = mock_summary
        response = await async_client.get("/api/v1/research/workflow/res_test_123/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["research_id"] == "res_test_123"
        assert data["sources_count"] == 4
        assert data["findings_count"] == 6

        # 2. Not found case
        mock_get.return_value = None
        response_404 = await async_client.get("/api/v1/research/workflow/unknown_id/summary")
        assert response_404.status_code == 404


@pytest.mark.asyncio
async def test_api_workflow_approve_success(async_client: AsyncClient):
    """Verify POST /api/v1/research/workflow/{id}/approve resumes workflow."""
    mock_resumed_state = {
        "research_id": "res_test_paused",
        "execution_status": WorkflowStatus.COMPLETED,
        "human_approved": True,
    }

    with patch(
        "app.api.v1.endpoints.research.workflow_service.approve_and_resume",
        new_callable=AsyncMock,
    ) as mock_resume:
        mock_resume.return_value = mock_resumed_state

        response = await async_client.post("/api/v1/research/workflow/res_test_paused/approve")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["research_id"] == "res_test_paused"
        assert data["status"] == "completed"
