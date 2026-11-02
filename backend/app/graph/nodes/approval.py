"""
Human-in-the-Loop Approval Node & Routing Logic.
Allows users to pause the workflow after the planning phase, inspect tasks,
and approve or modify the plan before autonomous researcher execution.
"""

from typing import Any, Dict
from app.core.logging import get_logger
from app.graph.state import AgentState
from app.schemas.events import ExecutionLogEvent, WorkflowStatus

logger = get_logger("graph.approval")


async def human_approval_node(state: AgentState) -> Dict[str, Any]:
    """
    Approval checkpoint node.
    Inspects human approval flag and logs transition.
    """
    research_id = state.get("research_id", "unknown")
    is_approved = state.get("human_approved", False)

    if is_approved:
        logger.info(f"[APPROVAL] Plan approved by user for research_id={research_id}. Proceeding to research.")
        return {
            "execution_status": WorkflowStatus.RESEARCHING,
            "current_step": "researcher_dispatch",
            "execution_logs": [
                ExecutionLogEvent(
                    research_id=research_id,
                    agent_name="orchestrator",
                    event_type="transition",
                    message="Human approval confirmed. Dispatching parallel Researcher agents.",
                )
            ],
        }

    logger.info(f"[APPROVAL] Research workflow {research_id} paused awaiting human plan approval.")
    return {
        "execution_status": WorkflowStatus.AWAITING_APPROVAL,
        "current_step": "human_approval",
        "execution_logs": [
            ExecutionLogEvent(
                research_id=research_id,
                agent_name="orchestrator",
                event_type="transition",
                message="Workflow paused awaiting human approval of research plan.",
            )
        ],
    }


def route_after_planning(state: AgentState) -> str:
    """
    Conditional router after the Planner node completes.
    Determines whether to pause for approval or proceed directly to research.
    """
    if state.get("execution_status") == WorkflowStatus.FAILED:
        return "error_node"

    if state.get("human_approval_required", False) and not state.get("human_approved", False):
        return "approval_node"

    return "researcher_dispatch"


def route_after_approval(state: AgentState) -> str:
    """
    Conditional router after the Approval checkpoint node.
    """
    if state.get("human_approved", False):
        return "researcher_dispatch"
    return "pause_workflow"

