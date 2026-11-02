"""
Analyst Node for AgentX LangGraph Workflow.
Invokes the AnalystAgent to synthesize cross-task patterns, trends, and contradictions
from collected research findings. Transitions the workflow to ANALYZING status.
"""

from typing import Any, Dict

from app.agents.analyst import analyst_agent
from app.core.logging import get_logger
from app.graph.state import AgentState
from app.schemas.events import ExecutionLogEvent, WorkflowStatus

logger = get_logger("graph.nodes.analyst")


async def analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Analyst.
    Processes all collected research findings into structured analytical insights.

    Transitions: RESEARCHING → ANALYZING → (next: fact_checker_node)
    """
    research_id = state.get("research_id", "unknown")
    findings = state.get("research_findings", [])
    tasks = state.get("research_tasks", [])
    query = state.get("user_query", "")

    logger.info(
        f"[ANALYST NODE] Starting analysis for research_id={research_id} | "
        f"findings={len(findings)} | tasks={len(tasks)}"
    )

    # Invoke the Analyst Agent
    try:
        payload = await analyst_agent.analyze_findings(
            findings=findings,
            tasks=tasks,
            query=query,
        )
    except Exception as exc:
        logger.error(f"[ANALYST NODE] AnalystAgent failed: {exc}", exc_info=True)
        return {
            "execution_status": WorkflowStatus.VERIFYING,
            "current_step": "analyst_error",
            "execution_logs": [
                ExecutionLogEvent(
                    research_id=research_id,
                    agent_name="analyst",
                    event_type="error",
                    message=f"Analyst agent failed: {exc}",
                    details={"error": str(exc)},
                )
            ],
        }

    # Persist insights from the payload
    insights = payload.insights
    key_patterns = payload.key_patterns
    contradictions = payload.contradictions_detected

    logger.info(
        f"[ANALYST NODE] Synthesis complete | "
        f"insights={len(insights)} | patterns={len(key_patterns)} | contradictions={len(contradictions)}"
    )

    return {
        "execution_status": WorkflowStatus.ANALYZING,
        "current_step": "analyst_complete",
        "analytical_insights": insights,
        "execution_logs": [
            ExecutionLogEvent(
                research_id=research_id,
                agent_name="analyst",
                event_type="artifact",
                message=(
                    f"Analyst synthesized {len(insights)} insights across {len(findings)} findings: "
                    f"{len(key_patterns)} emergent patterns | {len(contradictions)} contradictions detected."
                ),
                details={
                    "insights_count": len(insights),
                    "patterns": key_patterns,
                    "contradictions": contradictions,
                },
            )
        ],
    }

