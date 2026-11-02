"""
Unit tests for LangGraph state schema, initial state factory, and concurrency reducers.
"""

from app.graph.reducers import (
    reduce_findings,
    reduce_findings_by_task,
    reduce_logs,
    reduce_sources,
    reduce_token_usage,
)
from app.graph.state import create_initial_state
from app.schemas.events import ExecutionLogEvent, WorkflowStatus
from app.schemas.plan import ResearchDepth
from app.schemas.research import ResearchFinding, SourceMetadata


def test_create_initial_state():
    """Verify initial state factory generates well-formed state dictionary."""
    state = create_initial_state(
        query="The impact of AI on personalized medicine",
        depth=ResearchDepth.COMPREHENSIVE,
        human_approval_required=True,
    )

    assert state["research_id"].startswith("res_")
    assert state["user_query"] == "The impact of AI on personalized medicine"
    assert state["research_depth"] == ResearchDepth.COMPREHENSIVE
    assert state["human_approval_required"] is True
    assert state["human_approved"] is False
    assert state["execution_status"] == WorkflowStatus.INITIALIZING
    assert len(state["execution_logs"]) == 1
    assert state["execution_logs"][0].agent_name == "orchestrator"


def test_reduce_sources():
    """Verify source reducer deduplicates identical URLs and enriches content."""
    src1 = SourceMetadata(
        source_id="src_1",
        title="Nature AI Review",
        url="https://nature.com/article/1",
        domain="nature.com",
        snippet="Snippet 1",
        raw_content="Short text",
    )
    src2 = SourceMetadata(
        source_id="src_1",
        title="Nature AI Review",
        url="https://nature.com/article/1",
        domain="nature.com",
        snippet="Snippet 1",
        raw_content="Much longer and richer full text content for the article",
    )
    src3 = SourceMetadata(
        source_id="src_2",
        title="Science Mag AI",
        url="https://science.org/article/2",
        domain="science.org",
        snippet="Snippet 2",
    )

    # Merging src1 and src2 (same URL) should yield 1 source with the richer content
    merged_same = reduce_sources([src1], [src2])
    assert len(merged_same) == 1
    assert merged_same[0].raw_content == src2.raw_content

    # Merging distinct sources
    merged_all = reduce_sources(merged_same, [src3])
    assert len(merged_all) == 2


def test_reduce_findings():
    """Verify finding reducer deduplicates identical findings within a task."""
    fnd1 = ResearchFinding(
        finding_id="fnd_1",
        task_id="task_1",
        finding="AI detects nodules with 95% sensitivity",
        source_id="src_1",
        source_title="Title 1",
        source_url="https://example.com/1",
        supporting_excerpt="Passage 1",
    )
    fnd2 = ResearchFinding(
        finding_id="fnd_2",
        task_id="task_1",
        finding="AI detects nodules with 95% sensitivity",  # Duplicate claim in same task
        source_id="src_1",
        source_title="Title 1",
        source_url="https://example.com/1",
        supporting_excerpt="Passage 1",
    )
    fnd3 = ResearchFinding(
        finding_id="fnd_3",
        task_id="task_2",
        finding="Hospital EHR integration costs averaged $2.1M",
        source_id="src_2",
        source_title="Title 2",
        source_url="https://example.com/2",
        supporting_excerpt="Passage 2",
    )

    merged = reduce_findings([fnd1], [fnd2, fnd3])
    assert len(merged) == 2
    findings_texts = [f.finding for f in merged]
    assert "AI detects nodules with 95% sensitivity" in findings_texts
    assert "Hospital EHR integration costs averaged $2.1M" in findings_texts


def test_reduce_logs():
    """Verify log reducer appends events and ignores duplicate event IDs."""
    evt1 = ExecutionLogEvent(
        event_id="evt_1",
        research_id="res_test",
        agent_name="orchestrator",
        message="Started",
    )
    evt2 = ExecutionLogEvent(
        event_id="evt_2",
        research_id="res_test",
        agent_name="planner",
        message="Planned",
    )

    merged = reduce_logs([evt1], [evt1, evt2])
    assert len(merged) == 2
    assert [e.event_id for e in merged] == ["evt_1", "evt_2"]


def test_reduce_token_usage():
    """Verify token usage reducer sums numeric counters."""
    t1 = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    t2 = {"prompt_tokens": 200, "completion_tokens": 75, "total_tokens": 275}

    combined = reduce_token_usage(t1, t2)
    assert combined["prompt_tokens"] == 300
    assert combined["completion_tokens"] == 125
    assert combined["total_tokens"] == 425

