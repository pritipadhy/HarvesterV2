"""
DuckDuckGo Search Engine
========================

Free search engine with no API key required.

Uses the duckduckgo_search library for web search.

Requirements:
    - pip install duckduckgo-search

Usage:
    from harvester_v2.search.engines.duckduckgo import DuckDuckGoSearchEngine

    engine = DuckDuckGoSearchEngine()
    results = await engine.search("CISCO networking solutions", max_results=5)
"""

import asyncio
from typing import List, Dict, Any, Optional

from shared.logging import get_platform_logger
from ..base import BaseSearchEngine
from ..registry import register_search

logger = get_platform_logger(__name__)

# Try to import duckduckgo_search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    DDGS = None
    logger.warning("duckduckgo-search not installed. DuckDuckGoSearchEngine will be disabled.")


@register_search("duckduckgo")
class DuckDuckGoSearchEngine(BaseSearchEngine):
    """
    DuckDuckGo Search Engine.

    Free web search with no API key required.
    Good for general web search and news.

    Attributes:
        region: Search region (e.g., "wt-wt" for worldwide)
        safesearch: Safe search level ("on", "moderate", "off")
    """

    def __init__(self, region: str = "wt-wt", safesearch: str = "moderate"):
        """
        Initialize DuckDuckGo search engine.

        Args:
            region: Search region code
            safesearch: Safe search level
        """
        self.region = region
        self.safesearch = safesearch

        if not DDGS_AVAILABLE:
            logger.warning("DuckDuckGoSearchEngine initialized but duckduckgo-search not available")

    async def search(
        self,
        query: str,
        max_results: int = 5,
        time_range: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute DuckDuckGo search.

        Args:
            query: Search query string
            max_results: Maximum results to return
            time_range: Time filter ("d" = day, "w" = week, "m" = month, "y" = year)

        Returns:
            List of normalized search results
        """
        if not DDGS_AVAILABLE:
            logger.error("duckduckgo-search not installed, cannot execute search")
            return []

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._sync_search(query, max_results, time_range)
            )

            logger.info(
                "DuckDuckGo search completed",
                query=query[:50],
                results_count=len(results)
            )

            return results

        except Exception as e:
            logger.error("DuckDuckGo search failed", query=query[:50], error=str(e))
            return []

    def _sync_search(
        self,
        query: str,
        max_results: int,
        time_range: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Synchronous search implementation."""
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=time_range,
                max_results=max_results
            ))

        return self._normalize_results(query, raw_results)

    def _normalize_results(self, query: str, raw_results: List[Dict]) -> List[Dict[str, Any]]:
        """Normalize DuckDuckGo results to standard format."""
        normalized = []

        for r in raw_results:
            normalized.append({
                "query": query,
                "snippet": r.get("body", ""),
                "source": r.get("href", r.get("link", "")),
                "title": r.get("title", ""),
                "score": None,
                "engine": "duckduckgo"
            })

        return normalized

    async def news_search(
        self,
        query: str,
        max_results: int = 5,
        time_range: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search news articles.

        Args:
            query: Search query
            max_results: Maximum results
            time_range: Time filter

        Returns:
            List of news article results
        """
        if not DDGS_AVAILABLE:
            return []

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._sync_news_search(query, max_results, time_range)
            )
            return results
        except Exception as e:
            logger.error("DuckDuckGo news search failed", error=str(e))
            return []

    def _sync_news_search(
        self,
        query: str,
        max_results: int,
        time_range: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Synchronous news search."""
        with DDGS() as ddgs:
            raw_results = list(ddgs.news(
                query,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=time_range,
                max_results=max_results
            ))

        normalized = []
        for r in raw_results:
            normalized.append({
                "query": query,
                "snippet": r.get("body", ""),
                "source": r.get("url", ""),
                "title": r.get("title", ""),
                "date": r.get("date", ""),
                "engine": "duckduckgo_news"
            })

        return normalized
