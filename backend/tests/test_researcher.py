"""
Unit and integration tests for Researcher Agent, Research Coordinator, and Multi-Agent API.
Runs completely offline with isolated mocks.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient

from app.agents.planner import PlannerAgent
from app.agents.researcher import (
    ExtractedFindingItem,
    ExtractedFindingsPayload,
    ResearcherAgent,
)
from app.schemas.plan import (
    AgentType,
    ResearchDepth,
    ResearchPlan,
    ResearchTask,
    ResearchTaskType,
)
from app.schemas.research import (
    MultiAgentResearchResult,
    ResearchFinding,
    SearchResult,
    SourceMetadata,
)
from app.services.research_coordinator import ResearchCoordinator
from app.tools.source_manager import SourceManager


@pytest.fixture
def sample_task() -> ResearchTask:
    """Sample research task for testing researcher execution."""
    return ResearchTask(
        task_id="task_1",
        title="Current AI applications in healthcare",
        description="Investigate clinical AI, diagnostic radiology tools, and hospital operations automation.",
        research_type=ResearchTaskType.TECHNICAL,
        search_queries=["current clinical AI diagnostic applications 2024"],
        assigned_agents=[AgentType.RESEARCHER],
        priority=1,
    )


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Sample mock search results from Tavily."""
    return [
        SearchResult(
            title="FDA Approved AI in Radiology 2024",
            url="https://fda.gov/medical-devices/ai-radiology",
            content="Over 500 AI algorithms have been FDA cleared for medical imaging and radiology as of 2024.",
            published_date="2024-01-10",
        ),
        SearchResult(
            title="Hospital AI Adoption Trends",
            url="https://healthaffairs.org/ai-hospital-adoption",
            content="78% of large hospital networks report implementing at least one predictive AI model.",
            published_date="2024-02-18",
        ),
    ]


@pytest.mark.asyncio
async def test_researcher_agent_execute_task(
    sample_task: ResearchTask,
    sample_search_results: list[SearchResult],
):
    """Verify that ResearcherAgent executes search, enriches sources, and extracts findings."""
    mock_search_tool = AsyncMock()
    mock_search_tool.search.return_value = sample_search_results

    mock_llm = AsyncMock()
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = ExtractedFindingsPayload(
        findings=[
            ExtractedFindingItem(
                finding="Over 500 AI algorithms have been FDA cleared for radiology as of 2024.",
                source_url="https://fda.gov/medical-devices/ai-radiology",
                supporting_excerpt="Over 500 AI algorithms have been FDA cleared for medical imaging and radiology as of 2024.",
                confidence_score=0.95,
            )
        ]
    )

    agent = ResearcherAgent(search_tool=mock_search_tool)
    agent._structured_llm = mock_structured

    source_manager = SourceManager()
    output = await agent.execute_task(
        task=sample_task,
        source_manager=source_manager,
        max_sources_per_task=2,
    )

    assert output.task_id == "task_1"
    assert len(output.sources) == 2
    assert len(output.findings) == 1

    finding = output.findings[0]
    assert finding.task_id == "task_1"
    assert "500 AI algorithms" in finding.finding
    assert finding.source_url == "https://fda.gov/medical-devices/ai-radiology"
    assert finding.supporting_excerpt != ""
    assert finding.publication_date == "2024-01-10"


@pytest.mark.asyncio
async def test_coordinator_multi_agent_pipeline():
    """Verify ResearchCoordinator orchestrates Planner and parallel Researchers concurrently."""
    # 1. Mock Planner
    mock_planner = AsyncMock(spec=PlannerAgent)
    mock_planner.generate_plan.return_value = ResearchPlan(
        topic="AI in Healthcare",
        objective="Analyze AI in healthcare",
        research_depth=ResearchDepth.QUICK,
        research_tasks=[
            ResearchTask(
                task_id="task_1",
                title="Current AI applications",
                description="Investigate applications",
                search_queries=["healthcare AI applications"],
            ),
            ResearchTask(
                task_id="task_2",
                title="Implementation challenges",
                description="Investigate challenges",
                search_queries=["healthcare AI challenges"],
            ),
        ],
    )

    # 2. Mock Researcher
    mock_researcher = AsyncMock(spec=ResearcherAgent)

    async def mock_execute(task, source_manager, max_sources_per_task):
        src = source_manager.register_search_result(
            SearchResult(
                title=f"Source for {task.title}",
                url=f"https://example.org/{task.task_id}",
                content=f"Content for {task.title}",
            )
        )
        return AsyncMock(
            task_id=task.task_id,
            task_title=task.title,
            findings=[
                ResearchFinding(
                    task_id=task.task_id,
                    finding=f"Finding for {task.title}",
                    source_id=src.source_id,
                    source_title=src.title,
                    source_url=src.url,
                    supporting_excerpt=f"Content for {task.title}",
                )
            ],
            sources=[src],
            queries_executed=task.search_queries,
            execution_time_ms=10.0,
        )

    mock_researcher.execute_task.side_effect = mock_execute

    coordinator = ResearchCoordinator(
        planner=mock_planner,
        researcher=mock_researcher,
        max_concurrency=2,
    )

    result = await coordinator.run_research_pipeline(
        query="Research AI in healthcare",
        depth=ResearchDepth.QUICK,
    )

    assert result.query == "Research AI in healthcare"
    assert len(result.plan.research_tasks) == 2
    assert len(result.all_sources) == 2
    assert len(result.all_findings) == 2
    assert "task_1" in result.findings_by_task
    assert "task_2" in result.findings_by_task
    assert result.total_execution_time_ms >= 0.0


@pytest.mark.asyncio
async def test_api_execute_endpoint_success(async_client: AsyncClient):
    """Integration test: POST /api/v1/research/execute orchestrates multi-agent research."""
    mock_result = MultiAgentResearchResult(
        query="Research AI in healthcare",
        plan=ResearchPlan(
            topic="AI in Healthcare",
            objective="Evaluate healthcare AI",
            research_depth=ResearchDepth.QUICK,
            research_tasks=[
                ResearchTask(
                    task_id="task_1",
                    title="Applications",
                    description="Study clinical applications",
                )
            ],
        ),
        all_sources=[
            SourceMetadata(
                source_id="src_123",
                title="FDA Clearance Data",
                url="https://fda.gov/ai",
                domain="fda.gov",
                snippet="FDA cleared algorithms",
            )
        ],
        all_findings=[
            ResearchFinding(
                task_id="task_1",
                finding="FDA has approved numerous algorithms.",
                source_id="src_123",
                source_title="FDA Clearance Data",
                source_url="https://fda.gov/ai",
                supporting_excerpt="FDA cleared algorithms",
            )
        ],
        findings_by_task={
            "task_1": [
                ResearchFinding(
                    task_id="task_1",
                    finding="FDA has approved numerous algorithms.",
                    source_id="src_123",
                    source_title="FDA Clearance Data",
                    source_url="https://fda.gov/ai",
                    supporting_excerpt="FDA cleared algorithms",
                )
            ]
        },
        total_execution_time_ms=120.5,
    )

    with patch(
        "app.api.v1.endpoints.research.research_coordinator.run_research_pipeline",
        new_callable=AsyncMock,
    ) as mock_coord:
        mock_coord.return_value = mock_result

        response = await async_client.post(
            "/api/v1/research/execute",
            json={
                "query": "Research the future of artificial intelligence in healthcare.",
                "depth": "quick",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Multi-agent research completed" in data["message"]
        assert len(data["result"]["all_sources"]) == 1
        assert len(data["result"]["all_findings"]) == 1
        assert data["result"]["all_sources"][0]["domain"] == "fda.gov"


@pytest.mark.asyncio
async def test_api_execute_endpoint_validation_error(async_client: AsyncClient):
    """Integration test: invalid payload raises 422 error."""
    response = await async_client.post("/api/v1/research/execute", json={"query": ""})
    assert response.status_code == 422

