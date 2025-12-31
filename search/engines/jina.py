"""
Jina AI Search Engine
=====================

Jina AI provides semantic search capabilities and content extraction.

Requirements:
    - API key from jina.ai
    - Set JINA_API_KEY environment variable

Usage:
    from harvester_v2.search.engines.jina import JinaSearchEngine

    engine = JinaSearchEngine()
    results = await engine.search("CISCO networking solutions", max_results=5)
"""

import os
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional

from shared.logging import get_platform_logger
from ..base import BaseSearchEngine
from ..registry import register_search

logger = get_platform_logger(__name__)


@register_search("jina")
class JinaSearchEngine(BaseSearchEngine):
    """
    Jina AI Search Engine.

    Provides semantic search and content understanding.

    Attributes:
        api_key: Jina API key
        base_url: Jina API endpoint
    """

    # Jina Reader API for content extraction
    READER_URL = "https://r.jina.ai/"
    # Jina Search API
    SEARCH_URL = "https://s.jina.ai/"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Jina search engine.

        Args:
            api_key: Jina API key (defaults to JINA_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")

        if not self.api_key:
            logger.warning("JINA_API_KEY not set. JinaSearchEngine may have limited functionality.")

    async def search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute Jina search.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of normalized search results
        """
        headers = {
            "Accept": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                # Jina Search API
                url = f"{self.SEARCH_URL}{query}"

                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(
                            "Jina search failed",
                            status=response.status,
                            query=query[:50]
                        )
                        return []

                    data = await response.json()

            logger.info(
                "Jina search completed",
                query=query[:50],
                results_count=len(data.get("data", []))
            )

            return self._normalize_results(query, data, max_results)

        except Exception as e:
            logger.error("Jina search failed", query=query[:50], error=str(e))
            return []

    def _normalize_results(
        self,
        query: str,
        raw_result: Dict[str, Any],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Normalize Jina results to standard format."""
        normalized = []

        results = raw_result.get("data", [])[:max_results]

        for r in results:
            normalized.append({
                "query": query,
                "snippet": r.get("content", r.get("description", "")),
                "source": r.get("url", ""),
                "title": r.get("title", ""),
                "score": r.get("score", None),
                "engine": "jina"
            })

        return normalized

    async def read_url(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a URL using Jina Reader.

        Args:
            url: URL to read and extract content from

        Returns:
            Dict with extracted content
        """
        headers = {
            "Accept": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                reader_url = f"{self.READER_URL}{url}"

                async with session.get(reader_url, headers=headers) as response:
                    if response.status != 200:
                        logger.error("Jina reader failed", status=response.status, url=url)
                        return {"url": url, "error": f"Status {response.status}"}

                    data = await response.json()

            return {
                "url": url,
                "title": data.get("data", {}).get("title", ""),
                "content": data.get("data", {}).get("content", ""),
                "description": data.get("data", {}).get("description", ""),
                "engine": "jina_reader"
            }

        except Exception as e:
            logger.error("Jina reader failed", url=url, error=str(e))
            return {"url": url, "error": str(e)}
