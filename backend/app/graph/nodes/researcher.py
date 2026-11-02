"""
Researcher Fan-Out & Worker Graph Nodes.
Implements parallel task execution using LangGraph's dynamic Send API.
"""

from typing import Any, Dict, List, Union
from langgraph.types import Send

from app.agents.researcher import researcher_agent
from app.core.logging import get_logger
from app.graph.state import AgentState
from app.schemas.events import ExecutionLogEvent
from app.schemas.plan import ResearchTask
from app.schemas.research import ResearcherTaskOutput
from app.tools.source_manager import SourceManager

logger = get_logger("graph.researcher")


def researcher_fanout_dispatcher(state: AgentState) -> Union[List[Send], str]:
    """
    Conditional edge function that fans out parallel Researcher agents using LangGraph's Send API.
    One worker is dispatched per non-overlapping research task.
    """
    research_id = state.get("research_id", "unknown")
    tasks = state.get("research_tasks", [])

    if not tasks:
        logger.warning(f"[RESEARCHER] No research tasks found for research_id={research_id}")
        return "aggregator_node"

    logger.info(f"[RESEARCHER] Fanning out {len(tasks)} parallel researcher workers for research_id={research_id}")

    sends = []
    for task in tasks:
        task_obj = task if isinstance(task, ResearchTask) else ResearchTask(**task)
        sends.append(
            Send(
                "researcher_worker",
                {
                    "research_id": research_id,
                    "task": task_obj.model_dump(),
                },
            )
        )
    return sends


async def researcher_worker_node(state_or_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Individual parallel worker node.
    Executes web retrieval and evidence extraction for a single assigned task.
    """
    research_id = state_or_payload.get("research_id", "unknown")
    raw_task = state_or_payload.get("task")

    if not raw_task:
        logger.error(f"[WORKER] Missing task in payload for research_id={research_id}")
        return {}

    task = raw_task if isinstance(raw_task, ResearchTask) else ResearchTask(**raw_task)
    logger.info(f"[WORKER] Researcher executing task='{task.task_id}' | title='{task.title}'")

    source_manager = SourceManager()
    output: ResearcherTaskOutput = await researcher_agent.execute_task(
        task=task,
        source_manager=source_manager,
    )

    log_event = ExecutionLogEvent(
        research_id=research_id,
        agent_name="researcher",
        event_type="info",
        message=f"Researcher collected {len(output.sources)} sources and {len(output.findings)} findings for '{task.title}'",
        details={
            "task_id": task.task_id,
            "sources_count": len(output.sources),
            "findings_count": len(output.findings),
            "latency_ms": output.execution_time_ms,
        },
    )

    # Return partial updates that flow directly into AgentState reducers
    return {
        "collected_sources": output.sources,
        "research_findings": output.findings,
        "findings_by_task": {task.task_id: output.findings},
        "search_queries_executed": output.queries_executed,
        "execution_logs": [log_event],
    }
