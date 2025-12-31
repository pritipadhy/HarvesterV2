"""
Base Classes for Search Engines and Extractors
===============================================

Ported from EnterpriseGPT (enterprise-gpt/src/adapter/search/base.py)

Provides abstract interfaces for:
- BaseSearchEngine: Search query execution
- BaseExtractor: URL content extraction
- SearchResult: Normalized result format
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """
    Normalized search result for downstream processing.

    All search engines return results in this format to enable
    consistent aggregation and deduplication.

    Attributes:
        query: Original search query
        snippet: Text excerpt from the result
        source: Canonical URL of the result
        title: Page title (optional)
        score: Relevance score (optional)
    """
    query: str
    snippet: str
    source: str = Field(..., description="Canonical URL of the result")
    title: Optional[str] = None
    score: Optional[float] = None

    class Config:
        extra = "allow"  # Allow provider-specific metadata keys


class BaseSearchEngine(ABC):
    """
    Interface for search engine wrappers.

    All search engines must implement the async search method.
    Results should be normalized to SearchResult format.

    Example:
        @register_search("my_engine")
        class MySearchEngine(BaseSearchEngine):
            async def search(self, query: str, **params) -> List[Dict[str, Any]]:
                # Implementation
                return [{"query": query, "snippet": "...", "source": "..."}]
    """

    @abstractmethod
    async def search(self, query: str, **params) -> List[Dict[str, Any]]:
        """
        Execute a search and return normalized result dicts.

        Args:
            query: Search query string
            **params: Engine-specific parameters (max_results, etc.)

        Returns:
            List of dicts with at least: query, snippet, source
        """
        pass

    def _normalize(self, raw_results: Any) -> List[Dict[str, Any]]:
        """
        Normalize engine-specific results to standard format.

        Override this method for custom normalization logic.

        Args:
            raw_results: Raw response from search API

        Returns:
            List of normalized result dicts
        """
        if isinstance(raw_results, dict) and "results" in raw_results:
            return [
                {
                    "query": r.get("query", ""),
                    "snippet": r.get("content", r.get("snippet", "")),
                    "source": r.get("url", r.get("source", "")),
                    "title": r.get("title", ""),
                    "score": r.get("score", None)
                }
                for r in raw_results["results"]
            ]
        elif isinstance(raw_results, list):
            return [
                {
                    "query": r.get("query", ""),
                    "snippet": r.get("content", r.get("snippet", "")),
                    "source": r.get("url", r.get("source", "")),
                    "title": r.get("title", ""),
                    "score": r.get("score", None)
                }
                for r in raw_results
            ]
        return []


class BaseExtractor(ABC):
    """
    Interface for URL content extractors.

    Extractors fetch and parse content from URLs, returning
    structured data for downstream processing.

    Example:
        @register_extractor("my_extractor")
        class MyExtractor(BaseExtractor):
            async def extract(self, url: str, **params) -> Dict[str, Any]:
                # Implementation
                return {"url": url, "htmldata": "...", "markdown": "..."}
    """

    @abstractmethod
    async def extract(self, url: str, **params) -> Dict[str, Any]:
        """
        Extract content from a URL.

        Args:
            url: URL to extract content from
            **params: Extractor-specific parameters

        Returns:
            Dict with at least: url, htmldata (or content/markdown)
        """
        pass
