"""
Unit tests for Web Search Tools: TavilySearchTool, WebContentExtractor, and SourceManager.
Runs isolated without network access or paid API keys.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.schemas.research import SearchResult
from app.tools.source_manager import (
    SourceManager,
    extract_domain,
    normalize_url,
    calculate_source_reliability,
)
from app.tools.tavily_search import TavilySearchTool
from app.tools.web_extractor import WebContentExtractor


def test_normalize_url():
    """Verify URL normalization removes tracking parameters and trailing slashes."""
    url1 = "https://www.Nature.COM/articles/d41586-024-00123-x/?utm_source=twitter&utm_medium=social#heading"
    url2 = "https://nature.com/articles/d41586-024-00123-x"
    assert normalize_url(url1) == url2

    url_with_slash = "https://example.org/path/"
    assert normalize_url(url_with_slash) == "https://example.org/path"

    empty = normalize_url("")
    assert empty == ""


def test_extract_domain():
    """Verify domain extraction and www stripping."""
    assert extract_domain("https://www.fda.gov/medical-devices/ai") == "fda.gov"
    assert extract_domain("https://arxiv.org/abs/2301.00001") == "arxiv.org"
    assert extract_domain("invalid-url") == "unknown-domain"


def test_source_reliability_scoring():
    """Verify high-authority domains receive elevated reliability scores."""
    assert calculate_source_reliability("fda.gov") == 0.95
    assert calculate_source_reliability("ncbi.nlm.nih.gov") == 0.95
    assert calculate_source_reliability("random-blog.com") == 0.85


def test_source_manager_deduplication():
    """Verify SourceManager merges duplicate URLs and preserves single source_id."""
    manager = SourceManager()

    res1 = SearchResult(
        title="FDA AI Guidance 2024",
        url="https://www.fda.gov/guidance?utm_source=rss",
        content="Initial snippet from first task query.",
    )
    res2 = SearchResult(
        title="FDA AI Devices",
        url="https://fda.gov/guidance",
        content="Secondary snippet from subsequent task query.",
    )

    src1 = manager.register_search_result(res1)
    src2 = manager.register_search_result(res2)

    # Must be recognized as the exact same canonical source
    assert src1.source_id == src2.source_id
    assert src1.url == "https://fda.gov/guidance"
    assert len(manager.get_all_sources()) == 1


def test_source_manager_query_tracking():
    """Verify that executed queries are tracked to prevent redundant search calls."""
    manager = SourceManager()
    assert not manager.is_query_executed("AI healthcare market analysis")

    manager.mark_query_executed("AI healthcare market analysis")
    assert manager.is_query_executed("AI healthcare market analysis")
    # Case-insensitive verification
    assert manager.is_query_executed("ai healthcare market analysis")


@pytest.mark.asyncio
async def test_tavily_search_tool_success():
    """Verify Tavily search tool parses results properly."""
    tool = TavilySearchTool(api_key="tvly-test-key")
    mock_client = AsyncMock()
    mock_client.search.return_value = {
        "results": [
            {
                "title": "Clinical AI in 2024",
                "url": "https://nejm.org/article/10.1056/ai2024",
                "content": "Clinical AI is being widely deployed across diagnostic radiology.",
                "score": 0.98,
                "published_date": "2024-03-15",
            }
        ]
    }
    tool._client = mock_client

    results = await tool.search(query="clinical AI 2024", max_results=3)

    assert len(results) == 1
    assert results[0].title == "Clinical AI in 2024"
    assert results[0].url == "https://nejm.org/article/10.1056/ai2024"
    assert results[0].published_date == "2024-03-15"
    assert "diagnostic radiology" in results[0].content


@pytest.mark.asyncio
async def test_tavily_search_tool_error_handling():
    """Verify Tavily search tool handles API failures gracefully without raising."""
    tool = TavilySearchTool(api_key="tvly-test-key")
    mock_client = AsyncMock()
    mock_client.search.side_effect = Exception("Rate limit exceeded")
    tool._client = mock_client

    results = await tool.search(query="failing query")
    assert results == []


def test_web_content_extractor_clean_html():
    """Verify HTML cleaning strips non-content tags and collapses whitespace."""
    extractor = WebContentExtractor()
    sample_html = """
    <html>
      <head><title>Test Page</title></head>
      <body>
        <nav><a href="/">Home</a></nav>
        <script>console.log("bad script");</script>
        <style>.body { color: red; }</style>
        <h1>Medical AI Advancements</h1>
        <p>Artificial intelligence shows 94% diagnostic accuracy in radiology.</p>
        <footer>Copyright 2024</footer>
      </body>
    </html>
    """
    cleaned = extractor.clean_html(sample_html)
    assert "console.log" not in cleaned
    assert "Copyright" not in cleaned
    assert "Medical AI Advancements" in cleaned
    assert "94% diagnostic accuracy" in cleaned
