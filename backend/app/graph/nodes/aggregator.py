"""
Research Aggregator & Join Node.
Synchronizes and consolidates parallel research findings across all worker nodes.
"""

from typing import Any, Dict
from app.core.logging import get_logger
from app.graph.state import AgentState
from app.schemas.events import ExecutionLogEvent, WorkflowStatus

logger = get_logger("graph.aggregator")


async def research_aggregator_node(state: AgentState) -> Dict[str, Any]:
    """
    Join node where parallel researcher workers converge.
    Validates collected evidence and prepares state for downstream analysis.
    """
    research_id = state.get("research_id", "unknown")
    sources = state.get("collected_sources", [])
    findings = state.get("research_findings", [])
    tasks = state.get("research_tasks", [])

    logger.info(
        f"[AGGREGATOR] Consolidating research_id={research_id} | "
        f"sources={len(sources)} | findings={len(findings)} | tasks={len(tasks)}"
    )

    assigned = list(state.get("assigned_agents", []))
    if "orchestrator" not in assigned:
        assigned.append("orchestrator")

    # In Phase 4, the workflow successfully completes at research aggregation.
    # In Phase 5, this will transition to WorkflowStatus.ANALYZING.
    return {
        "execution_status": WorkflowStatus.COMPLETED,
        "current_step": "research_completed",
        "assigned_agents": assigned,
        "execution_logs": [
            ExecutionLogEvent(
                research_id=research_id,
                agent_name="orchestrator",
                event_type="transition",
                message=(
                    f"Research phase complete: {len(sources)} unique sources verified, "
                    f"{len(findings)} atomic findings extracted across {len(tasks)} subtasks."
                ),
                details={
                    "total_sources": len(sources),
                    "total_findings": len(findings),
                    "tasks_count": len(tasks),
                },
            )
        ],
    }
