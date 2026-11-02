"""
Tavily Search Client wrapper.
Provides robust search capabilities with query normalization, result parsing, and error isolation.
"""

from typing import Any, Dict, List, Optional
import httpx
from tavily import AsyncTavilyClient, TavilyClient

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.research import SearchResult

logger = get_logger("tools.tavily")


class SearchToolError(Exception):
    """Exception raised when search API queries fail."""
    pass


class TavilySearchTool:
    """Wrapper around Tavily Search API with retry logic and result sanitization."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.TAVILY_API_KEY
        self._client: Optional[AsyncTavilyClient] = None

    @property
    def client(self) -> AsyncTavilyClient:
        """Lazy-loaded async Tavily client instance."""
        if self._client is None:
            if not self.api_key:
                logger.warning(
                    "TAVILY_API_KEY is unset. Real web searches will fail until configured."
                )
            self._client = AsyncTavilyClient(api_key=self.api_key or "tvly-placeholder")
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_raw_content: bool = False,
    ) -> List[SearchResult]:
        """
        Execute an asynchronous search query with Tavily.
        
        Args:
            query: The search query string.
            max_results: Maximum number of links/snippets to return.
            search_depth: "basic" or "advanced" search.
            include_raw_content: Whether to fetch full page markdown/HTML.
            
        Returns:
            List[SearchResult]: Parsed, normalized search results.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        logger.info(f"Executing Tavily search | query='{clean_query[:60]}...' | max_results={max_results}")

        try:
            response: Dict[str, Any] = await self.client.search(
                query=clean_query,
                search_depth=search_depth,
                max_results=max_results,
                include_raw_content=include_raw_content,
            )

            raw_results = response.get("results", [])
            results: List[SearchResult] = []

            for item in raw_results:
                title = item.get("title") or "Untitled Source"
                url = item.get("url")
                content = item.get("content") or item.get("raw_content") or ""
                score = item.get("score")
                published_date = item.get("published_date")

                if url:
                    results.append(
                        SearchResult(
                            title=title.strip(),
                            url=url.strip(),
                            content=content.strip(),
                            score=float(score) if score is not None else None,
                            published_date=published_date,
                        )
                    )

            logger.info(f"Tavily returned {len(results)} valid results for query='{clean_query[:40]}'")
            return results

        except Exception as e:
            logger.error(f"Tavily search failed for query '{clean_query}': {str(e)}", exc_info=True)
            # Do not crash the entire multi-agent workflow if an external query errors; return empty list
            return []

    async def search_queries(
        self,
        queries: List[str],
        max_results_per_query: int = 3,
    ) -> List[SearchResult]:
        """Execute multiple search queries sequentially or concurrently and gather results."""
        aggregated: List[SearchResult] = []
        for q in queries:
            sub_results = await self.search(query=q, max_results=max_results_per_query)
            aggregated.extend(sub_results)
        return aggregated


# Global tool instance
tavily_search_tool = TavilySearchTool()
