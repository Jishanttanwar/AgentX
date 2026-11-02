"""
Source Deduplication and Citation Manager.
Canonicalizes URLs, manages unique source IDs, assesses source domain authority,
and prevents redundant searches across multi-agent workflows.
"""

from hashlib import sha256
from typing import Dict, List, Optional, Set
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.core.logging import get_logger
from app.schemas.research import SearchResult, SourceMetadata

logger = get_logger("tools.source_manager")

# Tracking parameters to strip during normalization
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "ref",
    "source",
}

# High-reliability domain suffixes and domains
HIGH_AUTHORITY_DOMAINS = {
    ".gov",
    ".edu",
    "ncbi.nlm.nih.gov",
    "nature.com",
    "sciencedirect.com",
    "nejm.org",
    "thelancet.com",
    "ieee.org",
    "acm.org",
    "arxiv.org",
    "who.int",
    "fda.gov",
    "cdc.gov",
}


def normalize_url(raw_url: str) -> str:
    """
    Canonicalize a URL by lowercasing host, removing tracking parameters,
    and stripping trailing slashes.
    """
    if not raw_url:
        return ""

    try:
        parsed = urlparse(raw_url.strip())
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]

        # Filter out tracking query params
        query_pairs = parse_qsl(parsed.query)
        clean_query_pairs = [
            (k, v) for k, v in query_pairs if k.lower() not in TRACKING_PARAMS
        ]
        clean_query = urlencode(clean_query_pairs)

        # Normalize path
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        clean_url = urlunparse(
            (parsed.scheme.lower() or "https", netloc, path, parsed.params, clean_query, "")
        )
        return clean_url
    except Exception:
        return raw_url.strip()


def extract_domain(url: str) -> str:
    """Extract clean domain name from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or "unknown-domain"
    except Exception:
        return "unknown-domain"


def calculate_source_reliability(domain: str) -> float:
    """Estimate a reliability score based on domain authority."""
    domain_lower = domain.lower()
    for high_auth in HIGH_AUTHORITY_DOMAINS:
        if domain_lower.endswith(high_auth) or high_auth in domain_lower:
            return 0.95
    return 0.85


def generate_source_id(canonical_url: str) -> str:
    """Generate a deterministic 8-character ID for a canonical source URL."""
    hashed = sha256(canonical_url.encode("utf-8")).hexdigest()
    return f"src_{hashed[:8]}"


class SourceManager:
    """Manages source uniqueness, indexing, and lookup across research sessions."""

    def __init__(self):
        self._sources_by_url: Dict[str, SourceMetadata] = {}
        self._executed_queries: Set[str] = set()

    def is_query_executed(self, query: str) -> bool:
        """Check if a query has already been executed in this workflow."""
        return query.strip().lower() in self._executed_queries

    def mark_query_executed(self, query: str) -> None:
        """Record query execution to prevent duplication."""
        self._executed_queries.add(query.strip().lower())

    def register_search_result(
        self,
        result: SearchResult,
        raw_content: Optional[str] = None,
    ) -> SourceMetadata:
        """
        Deduplicate, normalize, and register a raw search result into a persistent SourceMetadata.
        """
        canonical_url = normalize_url(result.url)
        domain = extract_domain(canonical_url)

        if canonical_url in self._sources_by_url:
            existing = self._sources_by_url[canonical_url]
            # Update content if new richer content is provided
            if raw_content and not existing.raw_content:
                existing.raw_content = raw_content
            return existing

        source_id = generate_source_id(canonical_url)
        source = SourceMetadata(
            source_id=source_id,
            title=result.title,
            url=canonical_url,
            domain=domain,
            snippet=result.content,
            raw_content=raw_content or result.content,
            publication_date=result.published_date,
            reliability_score=calculate_source_reliability(domain),
        )

        self._sources_by_url[canonical_url] = source
        return source

    def register_many(
        self,
        results: List[SearchResult],
    ) -> List[SourceMetadata]:
        """Register multiple search results, returning unique SourceMetadata instances."""
        registered = []
        for res in results:
            source = self.register_search_result(res)
            if source not in registered:
                registered.append(source)
        return registered

    def get_all_sources(self) -> List[SourceMetadata]:
        """Return all unique sources registered so far."""
        return list(self._sources_by_url.values())

    def get_source_by_id(self, source_id: str) -> Optional[SourceMetadata]:
        """Find a source by its unique ID."""
        for src in self._sources_by_url.values():
            if src.source_id == source_id:
                return src
        return None
