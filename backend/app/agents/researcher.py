"""
Researcher Agent Module.
Executes targeted web research for individual tasks, retrieves and sanitizes source content,
and extracts verified empirical findings mapped to source provenance.
"""

import time
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import get_llm
from app.core.logging import get_logger
from app.schemas.plan import ResearchTask
from app.schemas.research import (
    ResearcherTaskInput,
    ResearcherTaskOutput,
    ResearchFinding,
    SourceMetadata,
)
from app.tools.source_manager import SourceManager
from app.tools.tavily_search import TavilySearchTool, tavily_search_tool
from app.tools.web_extractor import WebContentExtractor, web_content_extractor

logger = get_logger("agents.researcher")

RESEARCHER_EXTRACTOR_PROMPT = """You are an Empirical Evidence Extraction Specialist for AgentX.

Your objective is to extract atomic, factual findings from the provided source documents for a specific research task.

CRITICAL INVARIANTS:
1. ONLY extract claims and facts that are explicitly supported by the provided source texts.
2. DO NOT hallucinate, infer unsupported conclusions, or invent statistics.
3. Every finding MUST cite the corresponding source URL and provide the exact or closely matching supporting excerpt from the text.
4. Extract 3 to 6 high-value findings containing concrete numbers, technical architectures, regulatory facts, or empirical trends.
"""


class ExtractedFindingItem(BaseModel):
    """Internal model for structured extraction from LLM."""
    finding: str = Field(description="Substantive factual statement or metric")
    source_url: str = Field(description="Exact URL of the source providing this evidence")
    supporting_excerpt: str = Field(
        description="Exact quote or verbatim passage from the source verifying the finding"
    )
    confidence_score: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence that finding is backed by excerpt",
    )


class ExtractedFindingsPayload(BaseModel):
    """Payload envelope for structured evidence extraction."""
    findings: List[ExtractedFindingItem] = Field(
        default_factory=list,
        description="List of verified factual findings extracted from texts",
    )


class ResearcherAgent:
    """Specialized agent responsible for web discovery and evidence extraction for a task."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        search_tool: Optional[TavilySearchTool] = None,
        extractor: Optional[WebContentExtractor] = None,
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name or settings.OPENAI_MODEL or "gpt-4o"
        self._llm = llm or get_llm(model=self.model_name, temperature=0.1)
        self._search_tool = search_tool or tavily_search_tool
        self._web_extractor = extractor or web_content_extractor
        self._structured_llm = self._llm.with_structured_output(ExtractedFindingsPayload)

    async def execute_task(
        self,
        task: ResearchTask,
        source_manager: SourceManager,
        max_sources_per_task: int = 4,
    ) -> ResearcherTaskOutput:
        """
        Execute research for a single research task.
        
        Args:
            task: The research task containing queries and objectives.
            source_manager: Shared source registry to deduplicate sources and queries.
            max_sources_per_task: Maximum sources to process for this task.
            
        Returns:
            ResearcherTaskOutput: Collected sources and verified findings.
        """
        start_time = time.time()
        logger.info(f"Researcher starting task='{task.task_id}' | title='{task.title}'")

        queries = task.search_queries or [task.title]
        queries_to_run = []
        for q in queries:
            if not source_manager.is_query_executed(q):
                queries_to_run.append(q)
                source_manager.mark_query_executed(q)
            else:
                logger.debug(f"Skipping duplicate query: '{q}'")

        # Fallback if all queries were executed earlier
        if not queries_to_run and queries:
            queries_to_run = [queries[0]]

        # Step 1: Search external sources
        raw_search_results = []
        for query_str in queries_to_run:
            results = await self._search_tool.search(
                query=query_str,
                max_results=max_sources_per_task,
            )
            raw_search_results.extend(results)

        # Step 2: Register & Deduplicate sources in SourceManager
        registered_sources: List[SourceMetadata] = []
        for res in raw_search_results:
            source = source_manager.register_search_result(res)
            if source not in registered_sources:
                registered_sources.append(source)
            if len(registered_sources) >= max_sources_per_task:
                break

        # Step 3: If snippets are thin, enrich top source with full text
        for src in registered_sources[:2]:
            if src.raw_content is None or len(src.raw_content) < 150:
                page_text = await self._web_extractor.fetch_page_content(src.url)
                if page_text:
                    src.raw_content = page_text

        # Step 4: Extract factual findings with LLM using structured output
        findings: List[ResearchFinding] = []
        if registered_sources:
            context_blocks = []
            for src in registered_sources:
                content = src.raw_content or src.snippet or "No text available."
                context_blocks.append(
                    f"SOURCE URL: {src.url}\nSOURCE TITLE: {src.title}\nTEXT EXCERPT:\n{content[:2500]}\n---"
                )

            evidence_context = "\n\n".join(context_blocks)
            user_prompt = (
                f"RESEARCH TASK: {task.title}\n"
                f"OBJECTIVE: {task.description}\n\n"
                f"DOCUMENTS:\n{evidence_context}\n\n"
                f"Extract 3 to 6 verified findings matching the task objective with exact excerpts."
            )

            try:
                extraction: ExtractedFindingsPayload = await self._structured_llm.ainvoke(
                    [
                        SystemMessage(content=RESEARCHER_EXTRACTOR_PROMPT),
                        HumanMessage(content=user_prompt),
                    ]
                )

                # Map extracted items to full ResearchFinding models with source metadata
                url_to_source = {s.url: s for s in registered_sources}

                for item in extraction.findings:
                    matched_source = url_to_source.get(item.source_url)
                    if not matched_source and registered_sources:
                        # Fallback to first source if exact URL string differed slightly
                        matched_source = registered_sources[0]

                    if matched_source:
                        findings.append(
                            ResearchFinding(
                                task_id=task.task_id,
                                finding=item.finding.strip(),
                                source_id=matched_source.source_id,
                                source_title=matched_source.title,
                                source_url=matched_source.url,
                                publication_date=matched_source.publication_date,
                                supporting_excerpt=item.supporting_excerpt.strip(),
                                confidence_score=item.confidence_score,
                            )
                        )
            except Exception as e:
                logger.error(
                    f"Researcher evidence extraction error on task '{task.task_id}': {e}",
                    exc_info=True,
                )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Researcher completed task '{task.task_id}' in {duration_ms:.2f}ms | "
            f"sources={len(registered_sources)} | findings={len(findings)}"
        )

        return ResearcherTaskOutput(
            task_id=task.task_id,
            task_title=task.title,
            findings=findings,
            sources=registered_sources,
            queries_executed=queries_to_run,
            execution_time_ms=round(duration_ms, 2),
        )


# Global researcher agent instance
researcher_agent = ResearcherAgent()
