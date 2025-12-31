"""
Firecrawl Content Extractor
===========================

Ported from EnterpriseGPT (enterprise-gpt/src/adapter/search/firecrawl.py)

Firecrawl provides intelligent web scraping with:
- JavaScript rendering
- Clean markdown extraction
- Structured data extraction

Requirements:
    - API key from firecrawl.dev
    - Set FIRECRAWL_API_KEY environment variable
    - pip install firecrawl-py

Usage:
    from harvester_v2.search.extractors.firecrawl import FirecrawlExtractor

    extractor = FirecrawlExtractor()
    content = await extractor.extract("https://example.com")
"""

import os
import asyncio
from typing import Dict, Any, Optional, List

from shared.logging import get_platform_logger
from ..base import BaseExtractor
from ..registry import register_extractor

logger = get_platform_logger(__name__)

# Try to import firecrawl
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    FirecrawlApp = None
    logger.warning("firecrawl-py not installed. FirecrawlExtractor will be disabled.")


@register_extractor("firecrawl")
class FirecrawlExtractor(BaseExtractor):
    """
    Firecrawl Content Extractor.

    Intelligent web scraping with JavaScript rendering and
    clean content extraction.

    Attributes:
        api_key: Firecrawl API key
        client: Firecrawl client instance
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Firecrawl extractor.

        Args:
            api_key: Firecrawl API key (defaults to FIRECRAWL_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        self.client = None

        if FIRECRAWL_AVAILABLE and self.api_key:
            try:
                self.client = FirecrawlApp(api_key=self.api_key)
            except Exception as e:
                logger.error("Failed to initialize Firecrawl client", error=str(e))

    async def extract(
        self,
        url: str,
        formats: Optional[List[str]] = None,
        only_main_content: bool = True
    ) -> Dict[str, Any]:
        """
        Extract content from a URL.

        Args:
            url: URL to extract content from
            formats: Output formats (e.g., ["markdown", "html"])
            only_main_content: Extract only main content, skip navigation etc.

        Returns:
            Dict with extracted content
        """
        if not FIRECRAWL_AVAILABLE:
            logger.error("firecrawl-py not installed, cannot extract")
            return {"url": url, "error": "firecrawl-py not installed"}

        if not self.client:
            logger.error("Firecrawl client not initialized (missing API key?)")
            return {"url": url, "error": "Firecrawl client not initialized"}

        formats = formats or ["markdown"]

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._sync_scrape(url, formats, only_main_content)
            )

            logger.info("Firecrawl extraction completed", url=url[:50])

            return result

        except Exception as e:
            logger.error("Firecrawl extraction failed", url=url, error=str(e))
            return {"url": url, "error": str(e)}

    def _sync_scrape(
        self,
        url: str,
        formats: List[str],
        only_main_content: bool
    ) -> Dict[str, Any]:
        """Synchronous scrape implementation."""
        response = self.client.scrape_url(
            url,
            params={
                "formats": formats,
                "onlyMainContent": only_main_content
            }
        )

        return {
            "url": url,
            "markdown": response.get("markdown", ""),
            "html": response.get("html", ""),
            "htmldata": response.get("html", response.get("markdown", "")),
            "metadata": response.get("metadata", {}),
            "engine": "firecrawl"
        }

    async def crawl_site(
        self,
        url: str,
        limit: int = 10,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Crawl multiple pages from a site.

        Args:
            url: Starting URL
            limit: Maximum pages to crawl
            include_paths: Only crawl paths matching these patterns
            exclude_paths: Skip paths matching these patterns

        Returns:
            List of extracted page contents
        """
        if not FIRECRAWL_AVAILABLE or not self.client:
            return []

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._sync_crawl(url, limit, include_paths, exclude_paths)
            )
            return result
        except Exception as e:
            logger.error("Firecrawl site crawl failed", url=url, error=str(e))
            return []

    def _sync_crawl(
        self,
        url: str,
        limit: int,
        include_paths: Optional[List[str]],
        exclude_paths: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Synchronous crawl implementation."""
        params = {
            "limit": limit,
            "scrapeOptions": {"formats": ["markdown"]}
        }

        if include_paths:
            params["includePaths"] = include_paths
        if exclude_paths:
            params["excludePaths"] = exclude_paths

        response = self.client.crawl_url(url, params=params)

        pages = []
        for page in response.get("data", []):
            pages.append({
                "url": page.get("metadata", {}).get("sourceURL", url),
                "markdown": page.get("markdown", ""),
                "title": page.get("metadata", {}).get("title", ""),
                "engine": "firecrawl"
            })

        return pages
