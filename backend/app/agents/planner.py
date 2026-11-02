"""
Planner Agent Module.
Converts user research queries into comprehensive, structured execution plans
with granular, non-overlapping tasks, search queries, and agent assignments.
"""

import time
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings
from app.core.llm import get_llm
from app.core.logging import get_logger
from app.schemas.plan import (
    ResearchDepth,
    ResearchPlan,
    ResearchTask,
    ResearchTaskType,
    AgentType,
)

logger = get_logger("agents.planner")

PLANNER_SYSTEM_PROMPT = """You are the Lead Research Strategist & Planning Agent for AgentX, an autonomous research intelligence system.

Your primary mission is to deconstruct a user's complex research inquiry into an actionable, structured, and non-overlapping research plan.

GUIDELINES:
1. Deconstruct the inquiry into comprehensive, distinct facets (e.g., current landscape, technical architecture, empirical data, regulatory/ethical challenges, and future outlook).
2. Calibrate the number of research tasks according to the requested depth:
   - "quick": 2 to 3 highly focused subtasks.
   - "standard": 4 to 6 balanced, well-rounded subtasks.
   - "comprehensive": 7 to 10 granular, exhaustive subtasks.
3. For each research task:
   - Provide a concise, clear title.
   - Define a concrete description clarifying exactly what evidence, statistics, or facts need to be collected.
   - Assign the appropriate domain category (general, market, technical, regulatory, academic, clinical_or_empirical, comparative).
   - Generate 2-3 precise search queries tailored for search engines (avoid conversational fluff in queries).
   - Assign the required specialized agents (e.g., RESEARCHER, ANALYST, FACT_CHECKER).
   - Designate a priority (1 = highest / critical path, 2 = standard, 3 = supplementary).
4. Never generate vague or redundant tasks. Each task must have a distinct investigative purpose.
5. Produce strictly structured output adhering to the ResearchPlan schema.
"""


class PlannerAgent:
    """Agent responsible for query understanding, subtopic decomposition, and plan generation."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.model_name = model_name or settings.OPENAI_MODEL or "gpt-4o"
        self._llm = llm or get_llm(
            model=self.model_name,
            temperature=temperature if temperature is not None else 0.2,
        )
        self._structured_llm = self._llm.with_structured_output(ResearchPlan)

    async def generate_plan(
        self,
        query: str,
        depth: ResearchDepth = ResearchDepth.STANDARD,
        context: Optional[str] = None,
    ) -> ResearchPlan:
        """
        Generate a validated, structured research plan from a user research query.
        
        Args:
            query: The research question or topic.
            depth: Desired research depth level.
            context: Optional additional domain constraints or background.
            
        Returns:
            ResearchPlan: Validated Pydantic plan instance.
            
        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If LLM generation or parsing fails.
        """
        clean_query = query.strip() if query else ""
        if not clean_query or len(clean_query) < 3:
            raise ValueError("Research query must be at least 3 non-whitespace characters.")

        logger.info(
            f"Planner initiating research planning | depth={depth.value} | query='{clean_query[:80]}...'"
        )
        start_time = time.time()

        user_content = f"Research Query: {clean_query}\nRequested Depth: {depth.value.upper()}"
        if context:
            user_content += f"\nAdditional Context / Directives: {context}"

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            plan: ResearchPlan = await self._structured_llm.ainvoke(messages)
            
            # Post-process and ensure normalized sequential task IDs
            normalized_tasks = []
            for index, task in enumerate(plan.research_tasks, start=1):
                clean_task_id = f"task_{index}"
                task_data = task.model_dump()
                task_data["task_id"] = clean_task_id
                
                # Ensure at least Researcher is assigned
                if not task_data.get("assigned_agents"):
                    task_data["assigned_agents"] = [AgentType.RESEARCHER]
                    
                normalized_tasks.append(ResearchTask(**task_data))

            plan.research_tasks = normalized_tasks
            plan.research_depth = depth

            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Planner successfully synthesized plan | topic='{plan.topic}' | "
                f"tasks_count={len(plan.research_tasks)} | duration={duration_ms:.2f}ms"
            )
            return plan

        except Exception as e:
            logger.error(f"Planner failed to generate research plan: {str(e)}", exc_info=True)
            raise RuntimeError(f"Planner Agent execution error: {str(e)}") from e


# Singleton instance for general use
planner_agent = PlannerAgent()

