"""
Tavily AI-Enhanced Search Engine
================================

Ported from EnterpriseGPT (enterprise-gpt/src/adapter/search/tavily.py)

Tavily provides AI-enhanced search results with:
- Advanced content filtering
- Answer generation
- Domain inclusion/exclusion
- Image search capabilities

Requirements:
    - API key from tavily.com
    - Set TAVILY_API_KEY environment variable
    - pip install langchain-tavily

Usage:
    from harvester_v2.search.engines.tavily import TavilySearchEngine

    engine = TavilySearchEngine()
    results = await engine.search("CISCO networking solutions", max_results=5)
"""

import os
import asyncio
from typing import List, Dict, Any, Optional

from shared.logging import get_platform_logger
from ..base import BaseSearchEngine
from ..registry import register_search

logger = get_platform_logger(__name__)

# Try to import langchain-tavily
try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilySearch = None
    logger.warning("langchain-tavily not installed. TavilySearchEngine will be disabled.")


@register_search("tavily")
class TavilySearchEngine(BaseSearchEngine):
    """
    Tavily AI-Enhanced Search Engine.

    Provides advanced search with AI-powered result filtering and ranking.

    Attributes:
        api_key: Tavily API key (from env or passed directly)
        search_depth: "basic" or "advanced"
    """

    def __init__(self, api_key: Optional[str] = None, search_depth: str = "advanced"):
        """
        Initialize Tavily search engine.

        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
            search_depth: Search depth - "basic" or "advanced"
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.search_depth = search_depth

        if not TAVILY_AVAILABLE:
            logger.warning("TavilySearchEngine initialized but langchain-tavily not available")

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute Tavily search.

        Args:
            query: Search query string
            max_results: Maximum results to return
            include_answer: Include AI-generated answer summary
            include_raw_content: Include full page content
            include_domains: Only search these domains
            exclude_domains: Exclude these domains from results

        Returns:
            List of normalized search results
        """
        if not TAVILY_AVAILABLE:
            logger.error("langchain-tavily not installed, cannot execute search")
            return []

        if not self.api_key:
            logger.error("TAVILY_API_KEY not set")
            return []

        try:
            tool = TavilySearch(
                max_results=max_results,
                api_key=self.api_key,
                search_depth=self.search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_domains=include_domains,
                exclude_domains=exclude_domains
            )

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tool.invoke({"query": query})
            )

            logger.info(
                "Tavily search completed",
                query=query[:50],
                results_count=len(result.get("results", [])) if isinstance(result, dict) else 0
            )

            return self._normalize_results(query, result)

        except Exception as e:
            logger.error("Tavily search failed", query=query[:50], error=str(e))
            return []

    def _normalize_results(self, query: str, raw_result: Any) -> List[Dict[str, Any]]:
        """Normalize Tavily results to standard format."""
        if not isinstance(raw_result, dict):
            return []

        results = raw_result.get("results", [])
        normalized = []

        for r in results:
            normalized.append({
                "query": query,
                "snippet": r.get("content", ""),
                "source": r.get("url", ""),
                "title": r.get("title", ""),
                "score": r.get("score", None),
                "engine": "tavily"
            })

        # Include answer if present
        if raw_result.get("answer"):
            normalized.insert(0, {
                "query": query,
                "snippet": raw_result["answer"],
                "source": "tavily_answer",
                "title": "AI-Generated Answer",
                "score": 1.0,
                "engine": "tavily"
            })

        return normalized
