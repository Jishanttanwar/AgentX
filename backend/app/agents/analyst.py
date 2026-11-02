"""
Analyst Agent Module.
Synthesizes research findings across subtasks to identify cross-cutting trends,
quantitative patterns, and conflicting evidence without hallucination.
"""

import time
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import get_llm
from app.core.logging import get_logger
from app.schemas.intelligence import AnalyticalInsight, InsightCategory
from app.schemas.plan import ResearchTask
from app.schemas.research import ResearchFinding

logger = get_logger("agents.analyst")

ANALYST_SYSTEM_PROMPT = """You are the Principal Intelligence Analyst for AgentX.

Your objective is to analyze verified research findings across multiple subtasks, synthesizing them into actionable, high-level analytical insights.

CRITICAL INVARIANTS:
1. STRICT DISTINCTION BETWEEN EVIDENCE AND INTERPRETATION:
   - "empirical_evidence": Must contain only verifiable facts, metrics, dates, and quotes present in the findings.
   - "strategic_interpretation": Your analytical deduction, synthesis of patterns, or implications based strictly on that evidence.
2. ZERO FABRICATION: Never invent numbers, market share percentages, CAGRs, or facts.
3. CONTRADICTION DETECTION: Explicitly highlight any divergent or conflicting data points across sources.
4. SYNTHESIS OVER REPETITION: Look for emergent patterns that span across 2 or more research tasks.
5. Provide 3 to 6 impactful insights categorized as trend, pattern, statistic, contradiction, or strategic.
"""


class AnalyticalInsightsPayload(BaseModel):
    """Structured envelope for LLM-synthesized analytical insights."""

    insights: List[AnalyticalInsight] = Field(
        default_factory=list,
        description="Comprehensive cross-task analytical insights",
    )
    key_patterns: List[str] = Field(
        default_factory=list,
        description="Key emergent themes or recurring patterns across the research",
    )
    contradictions_detected: List[str] = Field(
        default_factory=list,
        description="Contradictory findings or unresolved disputes identified across sources",
    )


class AnalystAgent:
    """Specialized agent responsible for qualitative synthesis, pattern recognition, and trend analysis."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name or settings.OPENAI_MODEL or "gpt-4o"
        self._llm = llm or get_llm(model=self.model_name, temperature=0.2)
        self._structured_llm = self._llm.with_structured_output(AnalyticalInsightsPayload)

    async def analyze_findings(
        self,
        findings: List[ResearchFinding],
        tasks: Optional[List[ResearchTask]] = None,
        query: str = "",
    ) -> AnalyticalInsightsPayload:
        """
        Analyze and synthesize collected empirical findings into structured insights.
        
        Args:
            findings: List of verified research findings from Researcher agents.
            tasks: Research tasks context.
            query: Original user research query.
            
        Returns:
            AnalyticalInsightsPayload: Structured insights and identified patterns.
        """
        start_time = time.time()
        logger.info(f"Analyst processing {len(findings)} research findings for query='{query[:60]}'")

        if not findings:
            logger.warning("Analyst received empty findings list; returning empty payload.")
            return AnalyticalInsightsPayload(
                insights=[],
                key_patterns=["Insufficient empirical data to synthesize patterns."],
                contradictions_detected=[],
            )

        # Format findings for prompt context
        finding_blocks = []
        for index, fnd in enumerate(findings, start=1):
            finding_blocks.append(
                f"[{fnd.finding_id}] (Task: {fnd.task_id}) - {fnd.finding}\n"
                f"Source: {fnd.source_title} ({fnd.source_url})\n"
                f"Excerpt: \"{fnd.supporting_excerpt}\"\n"
            )

        context_text = "\n".join(finding_blocks[:50])  # Cap at top 50 findings to avoid prompt bloat

        task_titles = [f"{t.task_id}: {t.title}" for t in (tasks or [])]
        task_context = "\n".join(task_titles) if task_titles else "General Research"

        user_content = (
            f"OVERALL RESEARCH OBJECTIVE: {query}\n\n"
            f"RESEARCH SUBTASKS:\n{task_context}\n\n"
            f"COLLECTED EVIDENCE FINDINGS:\n{context_text}\n\n"
            f"Synthesize trends, statistics, patterns, and contradictions strictly from this evidence."
        )

        try:
            payload: AnalyticalInsightsPayload = await self._structured_llm.ainvoke(
                [
                    SystemMessage(content=ANALYST_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]
            )

            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Analyst synthesized {len(payload.insights)} insights and "
                f"{len(payload.contradictions_detected)} contradictions in {duration_ms:.2f}ms"
            )
            return payload

        except Exception as e:
            logger.error(f"Analyst synthesis error: {str(e)}", exc_info=True)
            return AnalyticalInsightsPayload(
                insights=[],
                key_patterns=[],
                contradictions_detected=[f"Analysis failed due to error: {str(e)}"],
            )


# Global analyst agent instance
analyst_agent = AnalystAgent()

