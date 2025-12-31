"""
Web Search Orchestrator
=======================

Ported from EnterpriseGPT (enterprise-gpt/src/adapter/search/orchestrator.py)

Orchestrates parallel search across multiple search engines and
optional content extraction.

Features:
- Parallel search across Tavily, DuckDuckGo, Jina
- Content extraction via Firecrawl
- Result deduplication and ranking
- Configurable per-engine settings

Usage:
    from harvester_v2.search import WebSearchOrchestrator

    orchestrator = WebSearchOrchestrator({
        "search_engines": ["tavily", "duckduckgo", "jina"],
        "max_results_per_engine": 5,
        "extractors": ["firecrawl"],
        "max_urls_to_extract": 3
    })

    results_text, urls = await orchestrator.orchestrated_search(
        "CISCO competitor analysis networking"
    )
"""

import asyncio
from typing import Dict, Any, List, Set, Tuple, Optional
from urllib.parse import urlparse

from shared.logging import get_platform_logger
from .registry import SEARCH_ENGINE_REGISTRY, EXTRACTOR_REGISTRY

logger = get_platform_logger(__name__)


class WebSearchOrchestrator:
    """
    Multi-engine search orchestrator with parallel execution.

    Runs searches across multiple engines concurrently, merges results,
    and optionally extracts full content from top URLs.

    Attributes:
        engines: List of search engine names to use
        engine_configs: Per-engine configuration
        max_results: Max results per engine
        extractors: List of extractor names
        max_urls_to_extract: How many URLs to extract content from
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize search orchestrator.

        Args:
            config: Configuration dict with:
                - search_engines: List of engine names
                - search_engine_configurations: Per-engine params
                - max_results_per_engine: Max results per engine
                - extractors: List of extractor names
                - extractor_configurations: Per-extractor params
                - max_urls_to_extract: URLs to extract content from
        """
        self.engines = config.get("search_engines", ["tavily", "duckduckgo"])
        self.engine_configs = config.get("search_engine_configurations", {})
        self.max_results = config.get("max_results_per_engine", 5)
        self.extractors = config.get("extractors", [])
        self.extractor_configs = config.get("extractor_configurations", {})
        self.max_urls_to_extract = config.get("max_urls_to_extract", 3)

        # Initialize engine instances
        self._engine_instances: Dict[str, Any] = {}
        self._extractor_instances: Dict[str, Any] = {}

        self._initialize_engines()
        self._initialize_extractors()

    def _initialize_engines(self) -> None:
        """Initialize search engine instances."""
        for engine_name in self.engines:
            if engine_name.lower() not in SEARCH_ENGINE_REGISTRY:
                logger.warning(f"Unknown search engine: {engine_name}")
                continue

            engine_cls = SEARCH_ENGINE_REGISTRY[engine_name.lower()]
            engine_config = self.engine_configs.get(engine_name, {})
            params = engine_config.get("params", {})

            try:
                self._engine_instances[engine_name] = engine_cls(**params)
                logger.info(f"Initialized search engine: {engine_name}")
            except Exception as e:
                logger.error(f"Failed to initialize {engine_name}", error=str(e))

    def _initialize_extractors(self) -> None:
        """Initialize extractor instances."""
        for extractor_name in self.extractors:
            if extractor_name.lower() not in EXTRACTOR_REGISTRY:
                logger.warning(f"Unknown extractor: {extractor_name}")
                continue

            extractor_cls = EXTRACTOR_REGISTRY[extractor_name.lower()]
            extractor_config = self.extractor_configs.get(extractor_name, {})
            params = extractor_config.get("params", {})

            try:
                self._extractor_instances[extractor_name] = extractor_cls(**params)
                logger.info(f"Initialized extractor: {extractor_name}")
            except Exception as e:
                logger.error(f"Failed to initialize {extractor_name}", error=str(e))

    async def orchestrated_search(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        extract_content: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Execute parallel search across engines.

        Args:
            query: Search query string
            engines: Specific engines to use (defaults to all configured)
            extract_content: Whether to extract full content from URLs

        Returns:
            Tuple of:
                - Formatted search results string
                - List of all discovered URLs
        """
        engines = engines or list(self._engine_instances.keys())

        logger.info(
            "Starting orchestrated search",
            query=query[:50],
            engines=engines
        )

        # Parallel search across engines
        search_tasks = [
            self._search_engine(engine_name, query)
            for engine_name in engines
            if engine_name in self._engine_instances
        ]

        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect all results and URLs
        all_results: List[Dict[str, Any]] = []
        all_urls: Set[str] = set()

        for result in search_results:
            if isinstance(result, Exception):
                logger.error("Search engine failed", error=str(result))
                continue
            if isinstance(result, list):
                all_results.extend(result)
                for r in result:
                    if r.get("source"):
                        all_urls.add(r["source"])

        # Deduplicate results by URL
        seen_urls: Set[str] = set()
        unique_results: List[Dict[str, Any]] = []
        for r in all_results:
            url = r.get("source", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Optional content extraction
        extracted_content: List[str] = []
        if extract_content and self.max_urls_to_extract > 0 and self._extractor_instances:
            urls_to_extract = list(all_urls)[:self.max_urls_to_extract]
            extracted_content = await self._extract_content(urls_to_extract)

        # Format results
        formatted = self._format_results(unique_results, extracted_content, query)

        logger.info(
            "Orchestrated search completed",
            query=query[:50],
            total_results=len(unique_results),
            urls_found=len(all_urls),
            extracted=len(extracted_content)
        )

        return formatted, list(all_urls)

    async def _search_engine(
        self,
        engine_name: str,
        query: str
    ) -> List[Dict[str, Any]]:
        """Execute search on a single engine."""
        engine = self._engine_instances.get(engine_name)
        if not engine:
            return []

        engine_config = self.engine_configs.get(engine_name, {})
        params = engine_config.get("params", {})
        max_results = params.get("max_results", self.max_results)

        try:
            results = await engine.search(query, max_results=max_results)
            logger.debug(
                f"Engine {engine_name} returned {len(results)} results",
                engine=engine_name
            )
            return results
        except Exception as e:
            logger.error(f"Engine {engine_name} failed", error=str(e))
            return []

    async def _extract_content(
        self,
        urls: List[str]
    ) -> List[str]:
        """Extract content from URLs using configured extractors."""
        if not urls or not self._extractor_instances:
            return []

        extracted: List[str] = []

        # Use first available extractor
        extractor_name = list(self._extractor_instances.keys())[0]
        extractor = self._extractor_instances[extractor_name]

        for url in urls:
            try:
                result = await extractor.extract(url)
                content = result.get("markdown", result.get("htmldata", result.get("content", "")))
                if content:
                    # Truncate long content
                    if len(content) > 5000:
                        content = content[:5000] + "...[truncated]"
                    extracted.append(f"Content from {url} (via {extractor_name}):\n{content}\n---\n")
            except Exception as e:
                logger.error(f"Failed to extract {url}", error=str(e))

        return extracted

    def _format_results(
        self,
        results: List[Dict[str, Any]],
        extracted_content: List[str],
        query: str
    ) -> str:
        """Format search results as readable text."""
        lines = [f"Search Results for: {query}\n", "=" * 50, ""]

        for i, r in enumerate(results, 1):
            lines.append(f"Result {i}:")
            lines.append(f"  Title: {r.get('title', 'N/A')}")
            lines.append(f"  Source: {r.get('source', 'N/A')}")
            lines.append(f"  Engine: {r.get('engine', 'unknown')}")
            lines.append(f"  Snippet: {r.get('snippet', 'N/A')[:300]}")
            lines.append("")

        if extracted_content:
            lines.append("\n" + "=" * 50)
            lines.append("Extracted Content:")
            lines.append("=" * 50 + "\n")
            lines.extend(extracted_content)

        return "\n".join(lines)

    async def search_with_context(
        self,
        query: str,
        context: str,
        engines: Optional[List[str]] = None
    ) -> Tuple[str, List[str]]:
        """
        Search with additional context for query expansion.

        Args:
            query: Base search query
            context: Additional context to include
            engines: Specific engines to use

        Returns:
            Tuple of formatted results and URLs
        """
        # Combine query with context for better results
        enhanced_query = f"{query} {context}"
        return await self.orchestrated_search(enhanced_query, engines)

    def get_available_engines(self) -> List[str]:
        """Return list of initialized search engines."""
        return list(self._engine_instances.keys())

    def get_available_extractors(self) -> List[str]:
        """Return list of initialized extractors."""
        return list(self._extractor_instances.keys())
