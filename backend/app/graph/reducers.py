"""
State Reducers for LangGraph.
Ensures thread-safe, concurrent state updates during parallel agent fan-out and joins.
"""

from typing import Dict, List, Optional, Union
from app.schemas.events import ExecutionLogEvent
from app.schemas.research import ResearchFinding, SourceMetadata


def reduce_sources(
    existing: Union[List[SourceMetadata], List[dict], None],
    new: Union[List[SourceMetadata], List[dict], None],
) -> List[SourceMetadata]:
    """
    Reducer for merging web sources collected across parallel agents.
    Deduplicates strictly by source URL and source_id.
    """
    combined: Dict[str, SourceMetadata] = {}

    for item in existing or []:
        src = item if isinstance(item, SourceMetadata) else SourceMetadata(**item)
        combined[src.url] = src

    for item in new or []:
        src = item if isinstance(item, SourceMetadata) else SourceMetadata(**item)
        if src.url not in combined:
            combined[src.url] = src
        else:
            # If incoming source has richer raw content, enrich the existing record
            curr = combined[src.url]
            if src.raw_content and (not curr.raw_content or len(src.raw_content) > len(curr.raw_content)):
                curr.raw_content = src.raw_content

    return list(combined.values())


def reduce_findings(
    existing: Union[List[ResearchFinding], List[dict], None],
    new: Union[List[ResearchFinding], List[dict], None],
) -> List[ResearchFinding]:
    """
    Reducer for accumulating factual claims and findings across parallel researchers.
    Deduplicates by finding_id or identical task_id + finding text.
    """
    seen_keys = set()
    result: List[ResearchFinding] = []

    for item in (existing or []) + (new or []):
        fnd = item if isinstance(item, ResearchFinding) else ResearchFinding(**item)
        dedupe_key = (fnd.task_id, fnd.finding.strip().lower())
        if dedupe_key not in seen_keys:
            seen_keys.add(dedupe_key)
            result.append(fnd)

    return result


def reduce_logs(
    existing: Union[List[ExecutionLogEvent], List[dict], None],
    new: Union[List[ExecutionLogEvent], List[dict], None],
) -> List[ExecutionLogEvent]:
    """
    Reducer for accumulating structured audit events.
    Appends new events and preserves chronological sequence.
    """
    result: List[ExecutionLogEvent] = []
    seen_ids = set()

    for item in (existing or []) + (new or []):
        evt = item if isinstance(item, ExecutionLogEvent) else ExecutionLogEvent(**item)
        if evt.event_id not in seen_ids:
            seen_ids.add(evt.event_id)
            result.append(evt)

    return result


def reduce_token_usage(
    existing: Optional[Dict[str, int]],
    new: Optional[Dict[str, int]],
) -> Dict[str, int]:
    """Reducer for accumulating token usage metrics across all LLM invocations."""
    totals = dict(existing or {})
    for key, val in (new or {}).items():
        totals[key] = totals.get(key, 0) + val
    return totals


def reduce_findings_by_task(
    existing: Optional[Dict[str, List[ResearchFinding]]],
    new: Optional[Dict[str, List[ResearchFinding]]],
) -> Dict[str, List[ResearchFinding]]:
    """Reducer for merging findings dictionaries indexed by task_id."""
    merged: Dict[str, List[ResearchFinding]] = dict(existing or {})
    for task_id, fnd_list in (new or {}).items():
        if task_id not in merged:
            merged[task_id] = list(fnd_list)
        else:
            merged[task_id] = reduce_findings(merged[task_id], fnd_list)
    return merged
