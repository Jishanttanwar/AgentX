"""
Unit tests for the Analyst Agent.
All LLM calls are mocked — no API keys required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.analyst import AnalystAgent, AnalyticalInsightsPayload
from app.schemas.intelligence import AnalyticalInsight, InsightCategory
from app.schemas.research import ResearchFinding
from app.schemas.plan import ResearchTask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_finding(task_id: str = "task_1", idx: int = 1) -> ResearchFinding:
    return ResearchFinding(
        task_id=task_id,
        finding=f"AI adoption in healthcare has grown by {idx * 20}% annually.",
        source_id=f"src_{idx:04d}",
        source_title=f"HealthTech Report {idx}",
        source_url=f"https://healthtech.example.com/report-{idx}",
        supporting_excerpt=f"Annual growth rate of {idx * 20}% recorded in AI adoption.",
    )


def make_task(task_id: str = "task_1", title: str = "AI Healthcare") -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        title=title,
        description="Research AI applications in healthcare.",
        research_type="market",
    )


def make_mock_payload(n_insights: int = 3) -> AnalyticalInsightsPayload:
    return AnalyticalInsightsPayload(
        insights=[
            AnalyticalInsight(
                title=f"Insight {i}",
                category=InsightCategory.TREND,
                empirical_evidence=f"Key evidence {i} from sources.",
                strategic_interpretation=f"Interpretation {i}",
                related_task_ids=["task_1"],
            )
            for i in range(1, n_insights + 1)
        ],
        key_patterns=["Pattern A", "Pattern B"],
        contradictions_detected=["Conflict X"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyst_agent_synthesizes_findings():
    """Analyst agent should return structured insights when findings are present."""
    findings = [make_finding("task_1", i) for i in range(1, 5)]
    tasks = [make_task("task_1", "AI Healthcare Market")]

    mock_payload = make_mock_payload(n_insights=3)

    with patch.object(AnalystAgent, "__init__", lambda self, *a, **kw: None):
        agent = AnalystAgent.__new__(AnalystAgent)
        agent.model_name = "gpt-4o"
        agent._structured_llm = AsyncMock(return_value=mock_payload)
        agent._structured_llm.ainvoke = AsyncMock(return_value=mock_payload)

        result = await agent.analyze_findings(
            findings=findings,
            tasks=tasks,
            query="AI in healthcare future",
        )

    assert isinstance(result, AnalyticalInsightsPayload)
    assert len(result.insights) == 3
    assert result.insights[0].category == InsightCategory.TREND
    assert len(result.key_patterns) == 2
    assert len(result.contradictions_detected) == 1


@pytest.mark.asyncio
async def test_analyst_agent_empty_findings():
    """Analyst agent should gracefully handle empty findings list."""
    with patch.object(AnalystAgent, "__init__", lambda self, *a, **kw: None):
        agent = AnalystAgent.__new__(AnalystAgent)
        agent.model_name = "gpt-4o"
        agent._structured_llm = MagicMock()

        result = await agent.analyze_findings(
            findings=[],
            tasks=[],
            query="Empty query",
        )

    assert isinstance(result, AnalyticalInsightsPayload)
    assert result.insights == []
    assert "Insufficient empirical data" in result.key_patterns[0]


@pytest.mark.asyncio
async def test_analyst_agent_handles_llm_error():
    """Analyst agent should return empty payload with error message on LLM failure."""
    findings = [make_finding("task_1", 1)]

    with patch.object(AnalystAgent, "__init__", lambda self, *a, **kw: None):
        agent = AnalystAgent.__new__(AnalystAgent)
        agent.model_name = "gpt-4o"
        agent._structured_llm = MagicMock()
        agent._structured_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API failure"))

        result = await agent.analyze_findings(
            findings=findings,
            tasks=[],
            query="Failing query",
        )

    assert isinstance(result, AnalyticalInsightsPayload)
    assert result.insights == []
    assert any("Analysis failed" in c for c in result.contradictions_detected)


@pytest.mark.asyncio
async def test_analyst_agent_caps_findings_at_50():
    """Analyst agent should silently cap findings at 50 to avoid prompt bloat."""
    findings = [make_finding("task_1", i) for i in range(1, 70)]  # 69 findings
    mock_payload = make_mock_payload(n_insights=2)

    with patch.object(AnalystAgent, "__init__", lambda self, *a, **kw: None):
        agent = AnalystAgent.__new__(AnalystAgent)
        agent.model_name = "gpt-4o"
        agent._structured_llm = MagicMock()
        agent._structured_llm.ainvoke = AsyncMock(return_value=mock_payload)

        result = await agent.analyze_findings(
            findings=findings,
            tasks=[],
            query="Large findings set",
        )

    assert isinstance(result, AnalyticalInsightsPayload)
    assert len(result.insights) == 2


@pytest.mark.asyncio
async def test_analyst_agent_insight_fields_populated():
    """Analyst agent insights should have all required fields populated."""
    findings = [make_finding("task_1", 1)]
    tasks = [make_task("task_1")]

    insight = AnalyticalInsight(
        title="Healthcare AI Growth",
        summary="Consistent annual growth observed.",
        category=InsightCategory.STATISTIC,
        confidence_score=0.9,
        empirical_evidence=["20% annual growth"],
        strategic_interpretation="Rapid adoption curve.",
        source_task_ids=["task_1"],
        contradicts_task_ids=[],
    )
    mock_payload = AnalyticalInsightsPayload(
        insights=[insight],
        key_patterns=["Growth pattern"],
        contradictions_detected=[],
    )

    with patch.object(AnalystAgent, "__init__", lambda self, *a, **kw: None):
        agent = AnalystAgent.__new__(AnalystAgent)
        agent.model_name = "gpt-4o"
        agent._structured_llm = MagicMock()
        agent._structured_llm.ainvoke = AsyncMock(return_value=mock_payload)

        result = await agent.analyze_findings(
            findings=findings,
            tasks=tasks,
            query="Healthcare AI trend",
        )

    assert result.insights[0].title == "Healthcare AI Growth"
    assert result.insights[0].confidence_score == 0.9
    assert result.insights[0].category == InsightCategory.STATISTIC
