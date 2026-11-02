"""
Additional Research Node for AgentX LangGraph Workflow.
Expands research for claims flagged by the Fact Checker as unsupported or conflicting.
Converts AdditionalResearchRequests into ResearchTasks and fans out to researcher_worker nodes.
"""

from typing import Any, Dict, List, Union

from langgraph.types import Send

from app.core.logging import get_logger
from app.graph.state import AgentState
from app.schemas.events import ExecutionLogEvent, WorkflowStatus
from app.schemas.plan import ResearchTask

logger = get_logger("graph.nodes.additional_research")


def additional_research_dispatcher(state: AgentState) -> Union[List[Send], str]:
    """
    Dispatcher: Converts AdditionalResearchRequests from the Fact Checker
    into parallel researcher_worker Send commands for targeted follow-up research.

    Returns END if no requests remain.
    """
    from langgraph.graph import END

    additional_requests = state.get("additional_research_requests", [])
    if not additional_requests:
        logger.info("[ADDITIONAL RESEARCH] No additional research requests; proceeding to writer.")
        return END

    sends: List[Send] = []
    for req in additional_requests:
        # Build one ResearchTask per query in each request
        for query_text in req.queries:
            task = ResearchTask(
                title=f"Additional Research: {query_text[:80]}",
                description=(
                    f"Follow-up research to verify claim [{req.target_claim_id}]: {req.reason}. "
                    f"Query: {query_text}"
                ),
                research_type="factual_verification",
                priority=req.priority,
                estimated_sources=3,
            )
            sends.append(Send("researcher_worker", {"task": task}))

    logger.info(
        f"[ADDITIONAL RESEARCH] Dispatching {len(sends)} targeted research workers "
        f"for {len(additional_requests)} unsupported claims."
    )
    return sends


async def additional_research_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Additional Research Gate.
    Increments the research_cycle_count and clears previous additional_research_requests
    before the fan-out dispatcher re-routes to researcher_worker nodes.
    """
    research_id = state.get("research_id", "unknown")
    cycle = state.get("research_cycle_count", 0) + 1
    requests = state.get("additional_research_requests", [])

    logger.info(
        f"[ADDITIONAL RESEARCH NODE] Initiating additional research cycle={cycle} | "
        f"requests={len(requests)}"
    )

    return {
        "research_cycle_count": cycle,
        "execution_status": WorkflowStatus.RESEARCHING,
        "current_step": f"additional_research_cycle_{cycle}",
        "additional_research_requests": [],  # Clear after dispatching
        "execution_logs": [
            ExecutionLogEvent(
                research_id=research_id,
                agent_name="orchestrator",
                event_type="transition",
                message=(
                    f"Initiating additional research cycle {cycle} for "
                    f"{len(requests)} unsupported claim(s)."
                ),
                details={
                    "cycle": cycle,
                    "unsupported_claims": len(requests),
                    "claim_ids": [r.target_claim_id for r in requests],
                },
            )
        ],
    }
