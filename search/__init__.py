"""
Multi-Engine Search Module for Harvester V2
============================================

Provides parallel search across multiple search engines:
- Tavily (AI-enhanced)
- DuckDuckGo (free, no API key)
- Jina (semantic search)
- Firecrawl (content extraction)

Usage:
    from harvester_v2.search import WebSearchOrchestrator

    orchestrator = WebSearchOrchestrator({
        "search_engines": ["tavily", "duckduckgo"],
        "max_results_per_engine": 5
    })

    results, urls = await orchestrator.orchestrated_search(
        "CISCO competitor analysis networking"
    )
"""

from .base import BaseSearchEngine, BaseExtractor, SearchResult
from .registry import SEARCH_ENGINE_REGISTRY, EXTRACTOR_REGISTRY, register_search, register_extractor
from .orchestrator import WebSearchOrchestrator

# Import engines to trigger registration
from .engines import tavily, duckduckgo, jina
from .extractors import firecrawl

__all__ = [
    "BaseSearchEngine",
    "BaseExtractor",
    "SearchResult",
    "WebSearchOrchestrator",
    "SEARCH_ENGINE_REGISTRY",
    "EXTRACTOR_REGISTRY",
    "register_search",
    "register_extractor",
]
