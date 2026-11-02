"""
Fact Checker Node for AgentX LangGraph Workflow.
Invokes the FactCheckerAgent to extract atomic factual claims from research findings
and rigorously verify each claim against collected source evidence.
Routes to additional research if unsupported claims are found.
"""

from typing import Any, Dict

from app.agents.fact_checker import fact_checker_agent
from app.core.logging import get_logger
from app.graph.state import AgentState
from app.schemas.events import ExecutionLogEvent, WorkflowStatus

logger = get_logger("graph.nodes.fact_checker")


async def fact_checker_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Fact Checker.
    Extracts atomic claims and verifies each against source evidence.
    If unsupported claims found and research cycles remain, flags for additional research.

    Transitions: ANALYZING → VERIFYING → (next: additional_research_node | writer_node)
    """
    research_id = state.get("research_id", "unknown")
    findings = state.get("research_findings", [])
    sources = state.get("collected_sources", [])
    research_cycle_count = state.get("research_cycle_count", 0)
    max_research_cycles = state.get("max_research_cycles", 2)

    logger.info(
        f"[FACT CHECKER NODE] Starting verification for research_id={research_id} | "
        f"findings={len(findings)} | sources={len(sources)} | cycle={research_cycle_count}"
    )

    try:
        result = await fact_checker_agent.run(
            findings=findings,
            sources=sources,
            max_claims=20,
        )
    except Exception as exc:
        logger.error(f"[FACT CHECKER NODE] FactCheckerAgent failed: {exc}", exc_info=True)
        return {
            "execution_status": WorkflowStatus.VERIFYING,
            "current_step": "fact_checker_error",
            "execution_logs": [
                ExecutionLogEvent(
                    research_id=research_id,
                    agent_name="fact_checker",
                    event_type="error",
                    message=f"Fact Checker agent failed: {exc}",
                    details={"error": str(exc)},
                )
            ],
        }

    summary = result.summary
    additional_requests = result.additional_requests

    # Determine whether to request additional research
    should_expand = (
        summary.needs_additional_research
        and len(additional_requests) > 0
        and research_cycle_count < max_research_cycles
    )

    logger.info(
        f"[FACT CHECKER NODE] Verification complete | "
        f"claims={summary.total_claims_checked} | verified={summary.verified_count} | "
        f"partial={summary.partially_supported_count} | unsupported={summary.unsupported_count} | "
        f"needs_additional_research={should_expand}"
    )

    update: Dict[str, Any] = {
        "execution_status": WorkflowStatus.VERIFYING,
        "current_step": "fact_checker_complete",
        "extracted_claims": result.claims,
        "verification_results": result.verification_results,
        "additional_research_requests": additional_requests if should_expand else [],
        "execution_logs": [
            ExecutionLogEvent(
                research_id=research_id,
                agent_name="fact_checker",
                event_type="artifact",
                message=(
                    f"Fact Checker verified {summary.total_claims_checked} claims | "
                    f"verified={summary.verified_count} | partial={summary.partially_supported_count} | "
                    f"conflicting={summary.conflicting_count} | unsupported={summary.unsupported_count} | "
                    f"verification_rate={summary.verification_rate:.1%}"
                ),
                details={
                    "total_claims": summary.total_claims_checked,
                    "verified": summary.verified_count,
                    "partially_supported": summary.partially_supported_count,
                    "conflicting": summary.conflicting_count,
                    "unsupported": summary.unsupported_count,
                    "verification_rate": summary.verification_rate,
                    "needs_additional_research": should_expand,
                    "additional_requests_count": len(additional_requests),
                },
            )
        ],
    }

    return update

