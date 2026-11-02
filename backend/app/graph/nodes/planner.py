"""
Planner Graph Node.
Converts the research query into structured subtasks within the LangGraph state machine.
"""

from typing import Any, Dict
from app.agents.planner import planner_agent
from app.core.logging import get_logger
from app.graph.state import AgentState
from app.schemas.events import ExecutionLogEvent, WorkflowStatus
from app.schemas.plan import ResearchDepth

logger = get_logger("graph.planner")


async def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node for research planning and subtask breakdown.
    """
    research_id = state.get("research_id", "unknown")
    query = state.get("user_query", "")
    depth = state.get("research_depth", ResearchDepth.STANDARD)
    context = state.get("additional_context")

    logger.info(f"[PLANNER] Generating plan for research_id={research_id} | depth={depth}")

    try:
        # Check if plan was already supplied
        existing_plan = state.get("research_plan")
        if existing_plan:
            plan = existing_plan
            logger.info(f"[PLANNER] Using existing research plan for topic='{plan.topic}'")
        else:
            plan = await planner_agent.generate_plan(query=query, depth=depth, context=context)

        tasks = plan.research_tasks
        assigned = list(state.get("assigned_agents", []))
        if "researcher" not in assigned:
            assigned.append("researcher")

        approval_needed = state.get("human_approval_required", False)
        next_status = (
            WorkflowStatus.AWAITING_APPROVAL
            if approval_needed and not state.get("human_approved", False)
            else WorkflowStatus.RESEARCHING
        )

        return {
            "research_plan": plan,
            "research_tasks": tasks,
            "assigned_agents": assigned,
            "execution_status": next_status,
            "current_step": "human_approval" if approval_needed else "researcher_dispatch",
            "execution_logs": [
                ExecutionLogEvent(
                    research_id=research_id,
                    agent_name="planner",
                    event_type="artifact",
                    message=f"Planner formulated '{plan.topic}' into {len(tasks)} non-overlapping subtasks.",
                    details={"topic": plan.topic, "tasks_count": len(tasks)},
                )
            ],
        }

    except Exception as e:
        logger.error(f"[PLANNER] Plan synthesis failed: {str(e)}", exc_info=True)
        return {
            "execution_status": WorkflowStatus.FAILED,
            "error_details": f"Planner failed: {str(e)}",
            "execution_logs": [
                ExecutionLogEvent(
                    research_id=research_id,
                    agent_name="planner",
                    event_type="error",
                    message=f"Plan formulation failed: {str(e)}",
                )
            ],
        }

