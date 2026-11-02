"""
AgentX Stateful LangGraph Workflow Builder.
Constructs, connects, and compiles the multi-agent state graph with checkpointing.

Workflow Topology (Phase 5):
  [START] → orchestrator → planner
                             |── (approval required?) ──→ approval ──→ [fan-out]
                             |── (direct) ──────────────────────────→ [fan-out]
                                                                           │
                                               ┌────────────────────────────────────────┐
                                               │         researcher_worker (×N)         │
                                               └───────────────┬────────────────────────┘
                                                               │
                                                          aggregator
                                                               │
                                                          analyst_node
                                                               │
                                                       fact_checker_node
                                                               │
                              ┌──────────────── needs_additional_research? ────────────────┐
                              │ YES (cycles < max)                          NO              │
                              ▼                                             ▼               │
                    additional_research_node                            [END]               │
                              │                                                             │
                     [fan-out researcher_workers]                                           │
                              │                                                             │
                         aggregator ─────────────────────────────────────────────────────┘
"""

from typing import List, Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.core.logging import get_logger
from app.graph.nodes.additional_research import (
    additional_research_dispatcher,
    additional_research_node,
)
from app.graph.nodes.aggregator import research_aggregator_node
from app.graph.nodes.analyst import analyst_node
from app.graph.nodes.approval import human_approval_node
from app.graph.nodes.fact_checker import fact_checker_node
from app.graph.nodes.orchestrator import orchestrator_node
from app.graph.nodes.planner import planner_node
from app.graph.nodes.researcher import (
    researcher_fanout_dispatcher,
    researcher_worker_node,
)
from app.graph.state import AgentState
from app.memory.checkpointer import get_checkpointer

logger = get_logger("graph.workflow")


def route_after_planner(state: AgentState) -> Union[List[Send], str]:
    """
    Routes from Planner:
    - If human approval is required and not yet granted → 'approval'
    - Otherwise → fans out to parallel 'researcher_worker' nodes or 'aggregator'
    """
    if state.get("human_approval_required", False) and not state.get("human_approved", False):
        return "approval"
    return researcher_fanout_dispatcher(state)


def route_after_approval(state: AgentState) -> Union[List[Send], str]:
    """
    Routes from Approval checkpoint:
    - If approved → fans out to parallel 'researcher_worker' nodes
    - If not approved → ends graph execution (awaiting external resumption)
    """
    if state.get("human_approved", False):
        return researcher_fanout_dispatcher(state)
    return END


def route_after_fact_check(state: AgentState) -> Union[List[Send], str]:
    """
    Routes from Fact Checker:
    - If additional research is needed AND cycles remain → additional_research_node
    - Otherwise → END (report generation handled in Phase 6)
    """
    additional_requests = state.get("additional_research_requests", [])
    research_cycle_count = state.get("research_cycle_count", 0)
    max_research_cycles = state.get("max_research_cycles", 2)

    if additional_requests and research_cycle_count < max_research_cycles:
        logger.info(
            f"[ROUTER] Routing to additional_research: {len(additional_requests)} unsupported claims "
            f"(cycle {research_cycle_count}/{max_research_cycles})"
        )
        return "additional_research"

    logger.info("[ROUTER] Fact check complete; no further research needed. Workflow complete.")
    return END


def route_after_additional_research(state: AgentState) -> Union[List[Send], str]:
    """
    Routes from Additional Research Gate Node:
    - Fans out new researcher_workers for follow-up queries
    - Falls back to END if no active requests remain
    """
    sends = additional_research_dispatcher(state)
    return sends if sends != END else END


def build_research_graph(
    checkpointer: Union[BaseCheckpointSaver, bool] = True,
) -> CompiledStateGraph:
    """
    Constructs and compiles the AgentX multi-agent research graph.

    Phase 5 extends Phase 4 with:
    - analyst_node: Cross-task pattern synthesis
    - fact_checker_node: Atomic claim extraction and evidence-backed verification
    - additional_research_node: Iterative research loop for unsupported claims
    """
    builder = StateGraph(AgentState)

    # ── Register Nodes ──────────────────────────────────────────────────────────
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("approval", human_approval_node)
    builder.add_node("researcher_worker", researcher_worker_node)
    builder.add_node("aggregator", research_aggregator_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("additional_research", additional_research_node)

    # ── Fixed Transitions ────────────────────────────────────────────────────────
    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", "planner")
    builder.add_edge("researcher_worker", "aggregator")
    builder.add_edge("aggregator", "analyst")
    builder.add_edge("analyst", "fact_checker")

    # ── Conditional Transitions ──────────────────────────────────────────────────
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        ["approval", "researcher_worker", "aggregator", END],
    )
    builder.add_conditional_edges(
        "approval",
        route_after_approval,
        ["researcher_worker", "aggregator", END],
    )
    builder.add_conditional_edges(
        "fact_checker",
        route_after_fact_check,
        ["additional_research", END],
    )
    builder.add_conditional_edges(
        "additional_research",
        route_after_additional_research,
        ["researcher_worker", END],
    )

    # ── Compile ──────────────────────────────────────────────────────────────────
    saver = get_checkpointer() if checkpointer is True else (checkpointer or None)
    compiled = builder.compile(checkpointer=saver)

    logger.info(
        "Compiled AgentX Phase 5 research graph: "
        "orchestrator → planner → [researchers] → aggregator → analyst → fact_checker → [END|loop]"
    )
    return compiled


# Pre-compiled global workflow instance
research_graph = build_research_graph()
