"""
Search Engine and Extractor Registry
=====================================

Ported from EnterpriseGPT (enterprise-gpt/src/adapter/search/registry.py)

Provides decorator-based registration for search engines and extractors.
Engines register themselves at import time using the @register_search decorator.

Usage:
    from harvester_v2.search.registry import register_search, SEARCH_ENGINE_REGISTRY

    @register_search("my_engine")
    class MySearchEngine(BaseSearchEngine):
        ...

    # Later, get registered engine
    engine_cls = SEARCH_ENGINE_REGISTRY.get("my_engine")
"""

from typing import Dict, Type, Callable

from .base import BaseSearchEngine, BaseExtractor


# Registry mappings - populated by decorators at import time
SEARCH_ENGINE_REGISTRY: Dict[str, Type[BaseSearchEngine]] = {}
EXTRACTOR_REGISTRY: Dict[str, Type[BaseExtractor]] = {}


def register_search(name: str) -> Callable[[Type[BaseSearchEngine]], Type[BaseSearchEngine]]:
    """
    Decorator to register a search engine under the given name.

    Args:
        name: Registry key (e.g., "tavily", "duckduckgo")

    Returns:
        Decorator function that registers the class

    Example:
        @register_search("tavily")
        class TavilySearchEngine(BaseSearchEngine):
            async def search(self, query: str, **params):
                ...
    """
    def _wrap(cls: Type[BaseSearchEngine]) -> Type[BaseSearchEngine]:
        SEARCH_ENGINE_REGISTRY[name.lower()] = cls
        return cls
    return _wrap


def register_extractor(name: str) -> Callable[[Type[BaseExtractor]], Type[BaseExtractor]]:
    """
    Decorator to register an extractor under the given name.

    Args:
        name: Registry key (e.g., "firecrawl", "jina")

    Returns:
        Decorator function that registers the class

    Example:
        @register_extractor("firecrawl")
        class FirecrawlExtractor(BaseExtractor):
            async def extract(self, url: str, **params):
                ...
    """
    def _wrap(cls: Type[BaseExtractor]) -> Type[BaseExtractor]:
        EXTRACTOR_REGISTRY[name.lower()] = cls
        return cls
    return _wrap


def get_search_engine(name: str) -> Type[BaseSearchEngine]:
    """
    Get a registered search engine by name.

    Args:
        name: Registry key

    Returns:
        Search engine class

    Raises:
        KeyError: If engine not registered
    """
    name_lower = name.lower()
    if name_lower not in SEARCH_ENGINE_REGISTRY:
        available = list(SEARCH_ENGINE_REGISTRY.keys())
        raise KeyError(f"Search engine '{name}' not registered. Available: {available}")
    return SEARCH_ENGINE_REGISTRY[name_lower]


def get_extractor(name: str) -> Type[BaseExtractor]:
    """
    Get a registered extractor by name.

    Args:
        name: Registry key

    Returns:
        Extractor class

    Raises:
        KeyError: If extractor not registered
    """
    name_lower = name.lower()
    if name_lower not in EXTRACTOR_REGISTRY:
        available = list(EXTRACTOR_REGISTRY.keys())
        raise KeyError(f"Extractor '{name}' not registered. Available: {available}")
    return EXTRACTOR_REGISTRY[name_lower]


def list_search_engines() -> list[str]:
    """Return list of registered search engine names."""
    return list(SEARCH_ENGINE_REGISTRY.keys())


def list_extractors() -> list[str]:
    """Return list of registered extractor names."""
    return list(EXTRACTOR_REGISTRY.keys())
