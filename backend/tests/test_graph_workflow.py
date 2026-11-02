"""
Integration tests for LangGraph state machine, nodes, conditional edges, and parallel execution.
Runs completely offline with isolated mocks.
"""

from unittest.mock import AsyncMock, patch
import pytest

from app.graph.nodes.approval import human_approval_node, route_after_approval, route_after_planning
from app.graph.nodes.orchestrator import orchestrator_node
from app.graph.nodes.planner import planner_node
from app.graph.state import AgentState, create_initial_state
from app.graph.workflow import build_research_graph
from app.memory.checkpointer import MemorySaver, reset_checkpointer
from app.schemas.events import WorkflowStatus
from app.schemas.plan import ResearchDepth, ResearchPlan, ResearchTask
from app.schemas.research import ResearchFinding, SearchResult, SourceMetadata
from app.services.workflow_service import WorkflowService


@pytest.fixture(autouse=True)
def clean_checkpointer():
    """Ensure checkpointer is clean before each test."""
    reset_checkpointer()
    yield
    reset_checkpointer()


@pytest.fixture
def mock_plan() -> ResearchPlan:
    """Sample research plan with 2 subtasks."""
    return ResearchPlan(
        topic="AI in Healthcare",
        objective="Analyze AI in healthcare",
        research_depth=ResearchDepth.QUICK,
        research_tasks=[
            ResearchTask(
                task_id="task_1",
                title="Current AI applications",
                description="Investigate clinical AI",
                search_queries=["clinical AI applications"],
            ),
            ResearchTask(
                task_id="task_2",
                title="Implementation challenges",
                description="Investigate adoption friction",
                search_queries=["healthcare AI barriers"],
            ),
        ],
    )


@pytest.mark.asyncio
async def test_orchestrator_node_success():
    """Verify orchestrator node validates query and transitions to planning."""
    state = create_initial_state(query="AI in Medical Imaging")
    result = await orchestrator_node(state)
    assert result["execution_status"] == WorkflowStatus.PLANNING
    assert result["current_step"] == "planner"
    assert len(result["execution_logs"]) == 1


@pytest.mark.asyncio
async def test_orchestrator_node_invalid_query():
    """Verify orchestrator node catches short query and marks failed."""
    state = create_initial_state(query="a")
    result = await orchestrator_node(state)
    assert result["execution_status"] == WorkflowStatus.FAILED
    assert "at least 3 characters" in result["error_details"]


@pytest.mark.asyncio
async def test_planner_node_execution(mock_plan: ResearchPlan):
    """Verify planner node injects tasks and updates agent assignment."""
    state = create_initial_state(query="AI in Healthcare")
    with patch(
        "app.graph.nodes.planner.planner_agent.generate_plan",
        new_callable=AsyncMock,
    ) as mock_gen:
        mock_gen.return_value = mock_plan
        result = await planner_node(state)

        assert result["research_plan"].topic == "AI in Healthcare"
        assert len(result["research_tasks"]) == 2
        assert "researcher" in result["assigned_agents"]
        assert result["execution_status"] == WorkflowStatus.RESEARCHING


def test_routing_after_planning():
    """Verify conditional edge routing decisions from planning."""
    state_normal: AgentState = {"human_approval_required": False, "human_approved": False}
    assert route_after_planning(state_normal) == "researcher_dispatch"

    state_approval: AgentState = {"human_approval_required": True, "human_approved": False}
    assert route_after_planning(state_approval) == "approval_node"

    state_failed: AgentState = {"execution_status": WorkflowStatus.FAILED}
    assert route_after_planning(state_failed) == "error_node"


@pytest.mark.asyncio
async def test_workflow_direct_execution_to_completion(mock_plan: ResearchPlan):
    """
    End-to-end integration test:
    Verify that the compiled LangGraph executes directly to COMPLETED when human approval is not required.
    """
    checkpointer = MemorySaver()
    graph = build_research_graph(checkpointer=checkpointer)

    # Mock Planner and Researcher calls
    with patch("app.graph.nodes.planner.planner_agent.generate_plan", new_callable=AsyncMock) as mock_p, \
         patch("app.graph.nodes.researcher.researcher_agent.execute_task", new_callable=AsyncMock) as mock_r:

        mock_p.return_value = mock_plan

        async def mock_execute(task, source_manager):
            src = source_manager.register_search_result(
                SearchResult(
                    title=f"Source for {task.title}",
                    url=f"https://example.org/{task.task_id}",
                    content="Evidence content",
                )
            )
            from app.schemas.research import ResearcherTaskOutput
            return ResearcherTaskOutput(
                task_id=task.task_id,
                task_title=task.title,
                findings=[
                    ResearchFinding(
                        task_id=task.task_id,
                        finding=f"Finding for {task.title}",
                        source_id=src.source_id,
                        source_title=src.title,
                        source_url=src.url,
                        supporting_excerpt="Evidence content",
                    )
                ],
                sources=[src],
                queries_executed=task.search_queries,
            )

        mock_r.side_effect = mock_execute

        service = WorkflowService(graph=graph, checkpointer=checkpointer)
        final_state = await service.start_workflow(
            query="AI in Healthcare Research",
            depth=ResearchDepth.QUICK,
            human_approval_required=False,
        )

        assert final_state["execution_status"] == WorkflowStatus.COMPLETED
        assert len(final_state["collected_sources"]) == 2
        assert len(final_state["research_findings"]) == 2
        assert final_state["current_step"] == "research_completed"

        # Verify state is persisted in checkpointer
        saved_state = await service.get_state(final_state["research_id"])
        assert saved_state is not None
        assert saved_state["execution_status"] == WorkflowStatus.COMPLETED


@pytest.mark.asyncio
async def test_workflow_human_approval_and_resumption(mock_plan: ResearchPlan):
    """
    Integration test:
    Verify workflow halts when human_approval_required=True,
    and resumes to completion when approved.
    """
    checkpointer = MemorySaver()
    graph = build_research_graph(checkpointer=checkpointer)

    with patch("app.graph.nodes.planner.planner_agent.generate_plan", new_callable=AsyncMock) as mock_p, \
         patch("app.graph.nodes.researcher.researcher_agent.execute_task", new_callable=AsyncMock) as mock_r:

        mock_p.return_value = mock_plan

        async def mock_execute(task, source_manager):
            from app.schemas.research import ResearcherTaskOutput
            src = source_manager.register_search_result(
                SearchResult(
                    title="Source",
                    url="https://example.org/source",
                    content="Evidence",
                )
            )
            return ResearcherTaskOutput(
                task_id=task.task_id,
                task_title=task.title,
                findings=[
                    ResearchFinding(
                        task_id=task.task_id,
                        finding="Verified claim",
                        source_id=src.source_id,
                        source_title=src.title,
                        source_url=src.url,
                        supporting_excerpt="Evidence",
                    )
                ],
                sources=[src],
            )

        mock_r.side_effect = mock_execute

        service = WorkflowService(graph=graph, checkpointer=checkpointer)

        # 1. Start workflow with human_approval_required=True
        paused_state = await service.start_workflow(
            query="Deep learning in oncology",
            depth=ResearchDepth.QUICK,
            human_approval_required=True,
        )

        assert paused_state["execution_status"] == WorkflowStatus.AWAITING_APPROVAL
        assert paused_state["current_step"] == "human_approval"
        rid = paused_state["research_id"]

        # 2. Inspect summary
        summary = await service.get_summary(rid)
        assert summary.status == WorkflowStatus.AWAITING_APPROVAL
        assert summary.human_approval_required is True

        # 3. Approve and resume
        resumed_state = await service.approve_and_resume(rid)
        assert resumed_state["execution_status"] == WorkflowStatus.COMPLETED
        assert resumed_state["human_approved"] is True
        assert len(resumed_state["research_findings"]) > 0

