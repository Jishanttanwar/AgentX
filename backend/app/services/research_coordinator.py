"""
Research Collaboration Coordinator Service.
Coordinates the first multi-agent interaction: Planner Agent generates structured tasks,
and parallel Researcher Agents concurrently retrieve evidence and populate shared state.
"""

import asyncio
import time
from typing import Dict, List, Optional

from app.agents.planner import PlannerAgent, planner_agent
from app.agents.researcher import ResearcherAgent, researcher_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.plan import ResearchDepth, ResearchPlan, ResearchTask
from app.schemas.research import (
    MultiAgentResearchResult,
    ResearcherTaskOutput,
    ResearchFinding,
    SourceMetadata,
)
from app.tools.source_manager import SourceManager

logger = get_logger("services.coordinator")


class ResearchCoordinator:
    """Coordinates collaborative execution between Planner and parallel Researcher agents."""

    def __init__(
        self,
        planner: Optional[PlannerAgent] = None,
        researcher: Optional[ResearcherAgent] = None,
        max_concurrency: Optional[int] = None,
    ):
        self.planner = planner or planner_agent
        self.researcher = researcher or researcher_agent
        self.max_concurrency = max_concurrency or settings.MAX_CONCURRENT_RESEARCHERS

    async def _execute_single_task_bounded(
        self,
        task: ResearchTask,
        source_manager: SourceManager,
        semaphore: asyncio.Semaphore,
    ) -> ResearcherTaskOutput:
        """Run researcher execution for a single task within a bounded concurrency slot."""
        async with semaphore:
            logger.info(f"Coordinator dispatching task='{task.task_id}' to Researcher Agent")
            return await self.researcher.execute_task(
                task=task,
                source_manager=source_manager,
                max_sources_per_task=settings.MAX_SEARCH_RESULTS_PER_TASK,
            )

    async def run_research_pipeline(
        self,
        query: str,
        depth: ResearchDepth = ResearchDepth.STANDARD,
        context: Optional[str] = None,
        predefined_plan: Optional[ResearchPlan] = None,
    ) -> MultiAgentResearchResult:
        """
        Execute full Planner -> Parallel Researcher multi-agent workflow.
        
        Args:
            query: User's research question.
            depth: Research depth (quick, standard, comprehensive).
            context: Additional domain context.
            predefined_plan: Optional pre-approved plan (for Human-in-the-loop workflows).
            
        Returns:
            MultiAgentResearchResult: Aggregated multi-agent state with plan, sources, and findings.
        """
        start_time = time.time()
        logger.info(f"ResearchCoordinator initiated | query='{query[:70]}' | concurrency={self.max_concurrency}")

        # Step 1: Planning Phase (Planner Agent)
        if predefined_plan:
            plan = predefined_plan
            logger.info(f"Coordinator using pre-existing research plan for topic='{plan.topic}'")
        else:
            plan = await self.planner.generate_plan(query=query, depth=depth, context=context)

        # Step 2: Shared State & Session Initialization
        source_manager = SourceManager()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Step 3: Parallel Researcher Phase
        logger.info(
            f"Coordinator launching {len(plan.research_tasks)} parallel researcher tasks "
            f"(bounded by max {self.max_concurrency} concurrent workers)"
        )

        task_coroutines = [
            self._execute_single_task_bounded(
                task=task,
                source_manager=source_manager,
                semaphore=semaphore,
            )
            for task in plan.research_tasks
        ]

        task_outputs: List[ResearcherTaskOutput] = await asyncio.gather(*task_coroutines)

        # Step 4: Accumulate shared state across all tasks
        all_findings: List[ResearchFinding] = []
        findings_by_task: Dict[str, List[ResearchFinding]] = {}

        for out in task_outputs:
            findings_by_task[out.task_id] = out.findings
            all_findings.extend(out.findings)

        all_sources: List[SourceMetadata] = source_manager.get_all_sources()
        total_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"ResearchCoordinator completed multi-agent execution in {total_time_ms:.2f}ms | "
            f"tasks={len(task_outputs)} | total_sources={len(all_sources)} | total_findings={len(all_findings)}"
        )

        return MultiAgentResearchResult(
            query=query,
            plan=plan,
            all_sources=all_sources,
            all_findings=all_findings,
            findings_by_task=findings_by_task,
            total_execution_time_ms=round(total_time_ms, 2),
        )


# Global coordinator instance
research_coordinator = ResearchCoordinator()
