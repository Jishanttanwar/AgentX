"""
AgentX Stateful LangGraph Workflow Builder.
Constructs, connects, and compiles the multi-agent state graph with checkpointing.
"""

from typing import List, Union
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.core.logging import get_logger
from app.graph.nodes.aggregator import research_aggregator_node
from app.graph.nodes.approval import human_approval_node
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
    - If human approval is required and not yet granted -> 'approval'
    - Otherwise -> fans out to parallel 'researcher_worker' nodes or 'aggregator'
    """
    if state.get("human_approval_required", False) and not state.get("human_approved", False):
        return "approval"
    return researcher_fanout_dispatcher(state)


def route_after_approval(state: AgentState) -> Union[List[Send], str]:
    """
    Routes from Approval checkpoint:
    - If approved -> fans out to parallel 'researcher_worker' nodes
    - If not approved -> ends graph execution (awaiting external resumption)
    """
    if state.get("human_approved", False):
        return researcher_fanout_dispatcher(state)
    return END


def build_research_graph(
    checkpointer: Union[BaseCheckpointSaver, bool] = True,
) -> CompiledStateGraph:
    """
    Constructs and compiles the AgentX multi-agent research graph.
    
    Workflow Topology:
    [START] -> orchestrator -> planner
                                 |-- (approval required?) --> approval -- (approved?) --> [fan-out]
                                 |-- (direct execution)  -------------------------------> [fan-out]
                                                                                             |
                                                                              +--------------+--------------+
                                                                              |                             |
                                                                       researcher_worker             researcher_worker
                                                                              |                             |
                                                                              +--------------+--------------+
                                                                                             |
                                                                                         aggregator -> [END]
    """
    builder = StateGraph(AgentState)

    # 1. Register Nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("approval", human_approval_node)
    builder.add_node("researcher_worker", researcher_worker_node)
    builder.add_node("aggregator", research_aggregator_node)

    # 2. Register Standard Transitions
    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", "planner")

    # 3. Register Conditional Transitions
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

    # 4. Join and Termination
    builder.add_edge("researcher_worker", "aggregator")
    builder.add_edge("aggregator", END)

    # 5. Compile with Checkpointer
    saver = get_checkpointer() if checkpointer is True else (checkpointer or None)
    compiled = builder.compile(checkpointer=saver)

    logger.info("Compiled AgentX research LangGraph workflow with checkpointing enabled")
    return compiled


# Pre-compiled global workflow instance
research_graph = build_research_graph()
