"""
Pydantic schemas for research sources, findings, and researcher inputs/outputs.
Ensures rigorous traceability of evidence, citations, and source provenance.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, HttpUrl
from app.schemas.plan import ResearchPlan


class SourceMetadata(BaseModel):
    """Normalized metadata for a verified web source."""

    source_id: str = Field(
        default_factory=lambda: f"src_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the source",
    )
    title: str = Field(..., description="Page or article title")
    url: str = Field(..., description="Full canonical source URL")
    domain: str = Field(..., description="Extracted domain name of the source")
    snippet: str = Field(default="", description="Snippet or brief description")
    raw_content: Optional[str] = Field(
        default=None,
        description="Cleaned extracted webpage text content",
    )
    publication_date: Optional[str] = Field(
        default=None,
        description="Published or updated date if available from metadata",
    )
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when content was retrieved",
    )
    reliability_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Estimated source reliability based on domain and publisher",
    )


class ResearchFinding(BaseModel):
    """An individual empirical claim, fact, or finding linked to a verified source."""

    finding_id: str = Field(
        default_factory=lambda: f"fnd_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the finding",
    )
    task_id: str = Field(..., description="ID of the research task that yielded this finding")
    finding: str = Field(..., description="Substantive factual claim, statistic, or observation")
    source_id: str = Field(..., description="Unique ID of the supporting source")
    source_title: str = Field(..., description="Title of the source publication")
    source_url: str = Field(..., description="Verifiable URL of the source")
    publication_date: Optional[str] = Field(
        default=None,
        description="Publication date of the source, if available",
    )
    retrieval_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the source evidence was gathered",
    )
    supporting_excerpt: str = Field(
        ...,
        description="Direct or supporting excerpt from the source text verifying the finding",
    )
    confidence_score: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence that the finding is backed by the source text",
    )


class SearchResult(BaseModel):
    """Raw search engine result item from Tavily or web retrieval."""

    title: str
    url: str
    content: str
    score: Optional[float] = None
    published_date: Optional[str] = None


class ResearcherTaskInput(BaseModel):
    """Input payload passed to a Researcher Agent instance."""

    task_id: str
    task_title: str
    task_description: str
    search_queries: List[str] = Field(default_factory=list)
    max_sources: int = Field(default=5, ge=1, le=10)


class ResearcherTaskOutput(BaseModel):
    """Structured output returned by a Researcher Agent for a single task."""

    task_id: str
    task_title: str
    findings: List[ResearchFinding] = Field(default_factory=list)
    sources: List[SourceMetadata] = Field(default_factory=list)
    queries_executed: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0


class MultiAgentResearchResult(BaseModel):
    """Aggregated state across Planner and Parallel Researcher agents."""

    query: str
    plan: ResearchPlan
    all_sources: List[SourceMetadata] = Field(default_factory=list)
    all_findings: List[ResearchFinding] = Field(default_factory=list)
    findings_by_task: Dict[str, List[ResearchFinding]] = Field(default_factory=dict)
    total_execution_time_ms: float = 0.0

