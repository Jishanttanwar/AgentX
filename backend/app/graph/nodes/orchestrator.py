"""
Orchestrator Graph Node.
Manages workflow entry, query validation, and transition into the planning phase.
"""

from typing import Any, Dict
from app.core.logging import get_logger
from app.graph.state import AgentState
from app.schemas.events import ExecutionLogEvent, WorkflowStatus

logger = get_logger("graph.orchestrator")


async def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """
    Entrypoint orchestrator node.
    Validates user query and prepares workflow for research planning.
    """
    research_id = state.get("research_id", "unknown")
    query = state.get("user_query", "").strip()

    logger.info(f"[ORCHESTRATOR] Initializing workflow {research_id} for query='{query[:60]}'")

    if not query or len(query) < 3:
        error_msg = "User research query must be at least 3 characters long."
        logger.error(f"[ORCHESTRATOR] {error_msg}")
        return {
            "execution_status": WorkflowStatus.FAILED,
            "error_details": error_msg,
            "execution_logs": [
                ExecutionLogEvent(
                    research_id=research_id,
                    agent_name="orchestrator",
                    event_type="error",
                    message=error_msg,
                )
            ],
        }

    return {
        "execution_status": WorkflowStatus.PLANNING,
        "current_step": "planner",
        "execution_logs": [
            ExecutionLogEvent(
                research_id=research_id,
                agent_name="orchestrator",
                event_type="transition",
                message="Orchestrator passed query to Planner Agent.",
                details={"status": WorkflowStatus.PLANNING.value},
            )
        ],
    }

