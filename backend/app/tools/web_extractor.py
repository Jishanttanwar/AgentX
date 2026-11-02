"""
Web Content Extractor & Sanitizer.
Retrieves and parses webpage HTML into clean, sanitized text, stripping scripts, ads, and markup.
"""

import re
from typing import Optional
from bs4 import BeautifulSoup
import httpx

from app.core.logging import get_logger

logger = get_logger("tools.extractor")

# Default headers to mimic a modern, well-behaved research bot
DEFAULT_HEADERS = {
    "User-Agent": (
        "AgentX-Research-Bot/1.0 (Autonomous Multi-Agent Intelligence System; "
        "+https://github.com/Jishanttanwar/AgentX)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Tags to strip entirely
STRIP_TAGS = [
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "button",
]


class WebContentExtractor:
    """Safely extracts and sanitizes textual content from web URLs."""

    def __init__(self, timeout_seconds: float = 8.0, max_char_limit: int = 15000):
        self.timeout = timeout_seconds
        self.max_char_limit = max_char_limit

    def clean_html(self, raw_html: str) -> str:
        """
        Sanitize and parse raw HTML into clean structured plain text.
        
        Args:
            raw_html: Raw HTML string.
            
        Returns:
            str: Cleaned plain text representation.
        """
        if not raw_html or not raw_html.strip():
            return ""

        try:
            soup = BeautifulSoup(raw_html, "html.parser")

            # Remove unwanted elements
            for tag in soup.find_all(STRIP_TAGS):
                tag.decompose()

            # Extract text
            text = soup.get_text(separator="\n", strip=True)

            # Normalize whitespace: collapse multiple blank lines into two
            text = re.sub(r"\n\s*\n+", "\n\n", text)
            # Collapse multiple spaces
            text = re.sub(r"[ \t]+", " ", text)

            # Truncate to safety limit
            if len(text) > self.max_char_limit:
                text = text[: self.max_char_limit] + "\n...[Content truncated for length]..."

            return text.strip()

        except Exception as e:
            logger.warning(f"Failed to clean HTML content: {e}")
            return ""

    async def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Asynchronously fetch and sanitize webpage content from a URL.
        
        Args:
            url: Target website URL.
            
        Returns:
            Optional[str]: Clean text content, or None if retrieval failed.
        """
        if not url or not url.startswith(("http://", "https://")):
            return None

        try:
            async with httpx.AsyncClient(
                headers=DEFAULT_HEADERS,
                timeout=self.timeout,
                follow_redirects=True,
                verify=False,  # Avoid failing on non-critical SSL chain quirks during research
            ) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    logger.debug(f"HTTP {response.status_code} when fetching {url}")
                    return None

                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    logger.debug(f"Skipping non-HTML content type '{content_type}' at {url}")
                    return None

                return self.clean_html(response.text)

        except httpx.TimeoutException:
            logger.debug(f"Timeout while fetching {url}")
            return None
        except Exception as e:
            logger.debug(f"Could not fetch {url}: {str(e)}")
            return None


# Global extractor tool instance
web_content_extractor = WebContentExtractor()
