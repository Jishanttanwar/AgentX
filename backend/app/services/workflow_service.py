"""
Workflow Execution Service.
Manages LangGraph invocation, checkpoint threading, human approval resumption,
and state introspection.
"""

from typing import Any, Dict, Optional
from app.core.logging import get_logger
from app.graph.state import AgentState, create_initial_state
from app.graph.workflow import research_graph
from app.memory.checkpointer import get_checkpointer
from app.schemas.events import WorkflowStateSummary, WorkflowStatus
from app.schemas.plan import ResearchDepth

logger = get_logger("services.workflow")


class WorkflowService:
    """Manages LangGraph workflow runs, sessions, and checkpointed state."""

    def __init__(self, graph=None, checkpointer=None):
        self.graph = graph or research_graph
        self.checkpointer = checkpointer or get_checkpointer()

    def _get_config(self, research_id: str) -> Dict[str, Any]:
        """Generate LangGraph configuration dict with checkpoint thread_id."""
        return {"configurable": {"thread_id": research_id}}

    async def start_workflow(
        self,
        query: str,
        research_id: Optional[str] = None,
        depth: ResearchDepth = ResearchDepth.STANDARD,
        context: Optional[str] = None,
        human_approval_required: bool = False,
    ) -> AgentState:
        """
        Initialize and run the research workflow.
        If human_approval_required is True, the graph halts after planning and awaits approval.
        """
        initial_state = create_initial_state(
            query=query,
            research_id=research_id,
            depth=depth,
            context=context,
            human_approval_required=human_approval_required,
        )
        rid = initial_state["research_id"]
        config = self._get_config(rid)

        logger.info(
            f"Starting workflow for research_id={rid} | query='{query[:60]}' | "
            f"approval_required={human_approval_required}"
        )

        final_state: AgentState = await self.graph.ainvoke(initial_state, config=config)
        return final_state

    async def approve_and_resume(self, research_id: str) -> AgentState:
        """
        Resume a paused workflow after user reviews and approves the research plan.
        """
        config = self._get_config(research_id)
        current_state_tuple = await self.graph.aget_state(config)

        if not current_state_tuple or not current_state_tuple.values:
            raise ValueError(f"Workflow session '{research_id}' not found in checkpoints.")

        current_values = dict(current_state_tuple.values)
        if current_values.get("execution_status") != WorkflowStatus.AWAITING_APPROVAL:
            logger.warning(
                f"Workflow '{research_id}' is in status '{current_values.get('execution_status')}', "
                f"expected AWAITING_APPROVAL."
            )

        # Update checkpointed state with approval flag
        logger.info(f"Resuming workflow for research_id={research_id} with human approval")
        await self.graph.aupdate_state(
            config,
            {
                "human_approved": True,
                "execution_status": WorkflowStatus.RESEARCHING,
            },
            as_node="approval",
        )

        # Resume execution from current checkpoint
        resumed_state: AgentState = await self.graph.ainvoke(None, config=config)
        return resumed_state

    async def get_state(self, research_id: str) -> Optional[AgentState]:
        """Retrieve the latest checkpointed state for a given research_id."""
        config = self._get_config(research_id)
        state_tuple = await self.graph.aget_state(config)
        if state_tuple and state_tuple.values:
            return state_tuple.values
        return None

    async def get_summary(self, research_id: str) -> Optional[WorkflowStateSummary]:
        """Get high-level summary and metrics for an active or completed workflow."""
        state = await self.get_state(research_id)
        if not state:
            return None

        sources = state.get("collected_sources", [])
        findings = state.get("research_findings", [])
        tasks = state.get("research_tasks", [])
        insights = state.get("analytical_insights", [])
        verification_results = state.get("verification_results", [])

        # Calculate verification rate
        verified = sum(
            1 for v in verification_results
            if hasattr(v, "status") and v.status in ("verified", "partially_supported")
        )
        verification_rate = verified / len(verification_results) if verification_results else 0.0

        return WorkflowStateSummary(
            research_id=research_id,
            status=state.get("execution_status", WorkflowStatus.INITIALIZING),
            query=state.get("user_query", ""),
            current_step=state.get("current_step", "unknown"),
            tasks_total=len(tasks),
            tasks_completed=len([t for t in tasks if t.task_id in state.get("findings_by_task", {})]),
            sources_count=len(sources),
            findings_count=len(findings),
            insights_count=len(insights),
            claims_checked=len(verification_results),
            verification_rate=round(verification_rate, 3),
            research_cycle_count=state.get("research_cycle_count", 0),
            human_approval_required=state.get("human_approval_required", False),
            error_details=state.get("error_details"),
        )


# Global workflow service instance
workflow_service = WorkflowService()

