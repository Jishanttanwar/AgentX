"""
Unit and integration tests for Planner Agent and Plan API endpoints.
Uses isolated mocks for LLM calls so tests never require paid API keys or network access.
"""

from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient

from app.agents.planner import PlannerAgent
from app.schemas.plan import (
    AgentType,
    ResearchDepth,
    ResearchPlan,
    ResearchTask,
    ResearchTaskType,
)


@pytest.fixture
def sample_healthcare_plan() -> ResearchPlan:
    """Fixture returning sample research plan matching user specifications."""
    return ResearchPlan(
        topic="AI in Healthcare",
        objective="Analyze current implementations, market trends, regulatory barriers, and future opportunities for AI in healthcare.",
        research_depth=ResearchDepth.STANDARD,
        research_tasks=[
            ResearchTask(
                task_id="task_1",
                title="Current AI applications in healthcare",
                description="Investigate clinical AI, diagnostic radiology tools, and hospital operations automation.",
                research_type=ResearchTaskType.TECHNICAL,
                search_queries=["current clinical AI diagnostic applications 2024", "AI hospital operations radiology"],
                assigned_agents=[AgentType.RESEARCHER],
                priority=1,
            ),
            ResearchTask(
                task_id="task_2",
                title="AI healthcare market analysis",
                description="Analyze global healthcare AI market size, CAGR, investment trends, and key enterprise players.",
                research_type=ResearchTaskType.MARKET,
                search_queries=["healthcare AI market size CAGR 2024 2030", "venture funding healthcare AI"],
                assigned_agents=[AgentType.RESEARCHER],
                priority=1,
            ),
            ResearchTask(
                task_id="task_3",
                title="Major technologies",
                description="Review foundational technologies including generative LLMs, computer vision, and predictive analytics.",
                research_type=ResearchTaskType.TECHNICAL,
                search_queries=["generative AI computer vision medical diagnostics", "predictive analytics in patient care"],
                assigned_agents=[AgentType.RESEARCHER],
                priority=2,
            ),
            ResearchTask(
                task_id="task_4",
                title="Implementation challenges",
                description="Examine integration friction with EHR systems, data privacy, model drift, and clinician adoption.",
                research_type=ResearchTaskType.GENERAL,
                search_queries=["EHR AI integration challenges", "clinician resistance healthcare AI adoption"],
                assigned_agents=[AgentType.RESEARCHER],
                priority=2,
            ),
            ResearchTask(
                task_id="task_5",
                title="Regulatory considerations",
                description="Evaluate FDA approval pathways, EU AI Act compliance for medical devices, and liability frameworks.",
                research_type=ResearchTaskType.REGULATORY,
                search_queries=["FDA guidance AI medical devices 2024", "EU AI Act medical device high risk"],
                assigned_agents=[AgentType.RESEARCHER],
                priority=1,
            ),
            ResearchTask(
                task_id="task_6",
                title="Future opportunities",
                description="Explore emerging horizons such as autonomous robot surgery, precision oncology, and personalized therapeutics.",
                research_type=ResearchTaskType.GENERAL,
                search_queries=["future robotics surgery AI therapeutics", "precision medicine oncology AI forecast"],
                assigned_agents=[AgentType.RESEARCHER],
                priority=3,
            ),
        ],
    )


def test_research_plan_schema_validation(sample_healthcare_plan: ResearchPlan):
    """Verify that ResearchPlan and computed fields behave correctly."""
    assert sample_healthcare_plan.topic == "AI in Healthcare"
    assert len(sample_healthcare_plan.research_tasks) == 6
    assert sample_healthcare_plan.task_titles == [
        "Current AI applications in healthcare",
        "AI healthcare market analysis",
        "Major technologies",
        "Implementation challenges",
        "Regulatory considerations",
        "Future opportunities",
    ]
    # Verify task 1 details
    task_1 = sample_healthcare_plan.research_tasks[0]
    assert task_1.task_id == "task_1"
    assert task_1.research_type == ResearchTaskType.TECHNICAL
    assert AgentType.RESEARCHER in task_1.assigned_agents


@pytest.mark.asyncio
async def test_planner_agent_generate_plan_success(sample_healthcare_plan: ResearchPlan):
    """Verify that PlannerAgent properly invokes structured LLM and normalizes output."""
    mock_llm = AsyncMock()
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = sample_healthcare_plan

    agent = PlannerAgent()
    agent._structured_llm = mock_structured

    plan = await agent.generate_plan(
        query="Research the future of artificial intelligence in healthcare.",
        depth=ResearchDepth.STANDARD,
    )

    assert plan.topic == "AI in Healthcare"
    assert len(plan.research_tasks) == 6
    assert plan.research_tasks[0].task_id == "task_1"
    assert plan.research_tasks[5].task_id == "task_6"
    assert mock_structured.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_planner_agent_invalid_empty_query():
    """Verify that empty or whitespace query raises ValueError without invoking LLM."""
    agent = PlannerAgent()
    with pytest.raises(ValueError, match="Research query must be at least 3"):
        await agent.generate_plan(query="  ")


@pytest.mark.asyncio
async def test_planner_agent_llm_failure_handling():
    """Verify that LLM network or API exceptions are caught and wrapped cleanly."""
    mock_structured = AsyncMock()
    mock_structured.ainvoke.side_effect = ConnectionError("OpenAI connection timeout")

    agent = PlannerAgent()
    agent._structured_llm = mock_structured

    with pytest.raises(RuntimeError, match="Planner Agent execution error"):
        await agent.generate_plan(query="Quantum Computing Trends 2025")


@pytest.mark.asyncio
async def test_api_generate_plan_endpoint_success(
    async_client: AsyncClient,
    sample_healthcare_plan: ResearchPlan,
):
    """Integration test: POST /api/v1/research/plan returns 200 with structured plan."""
    with patch(
        "app.api.v1.endpoints.research.planner_agent.generate_plan",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = sample_healthcare_plan

        payload = {
            "query": "Research the future of artificial intelligence in healthcare.",
            "depth": "standard",
            "additional_context": "Focus on clinical diagnostic adoption.",
        }
        response = await async_client.post("/api/v1/research/plan", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["plan"]["topic"] == "AI in Healthcare"
        assert len(data["plan"]["research_tasks"]) == 6
        assert data["plan"]["task_titles"] == [
            "Current AI applications in healthcare",
            "AI healthcare market analysis",
            "Major technologies",
            "Implementation challenges",
            "Regulatory considerations",
            "Future opportunities",
        ]
        assert data["execution_time_ms"] >= 0.0


@pytest.mark.asyncio
async def test_api_generate_plan_validation_error(async_client: AsyncClient):
    """Integration test: invalid request payload triggers validation error."""
    response = await async_client.post("/api/v1/research/plan", json={"query": "ab"})
    assert response.status_code == 422

