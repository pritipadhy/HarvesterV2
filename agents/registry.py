"""
Sub-Agent Auto-Discovery
========================

Provides automatic discovery of sub-agents from the agents/core/ directory.

NOTE: This is separate from SubAgentRegistry in parallel_executor.py which
handles manual @SubAgentRegistry.register decorator registration + tenant config.

This module provides filesystem-based auto-discovery that can:
1. Find all *_subagent.py files automatically
2. Feed discovered classes into SubAgentRegistry.register()

Features:
- Scans for *_subagent.py files
- Auto-discovers classes ending in "SubAgent"
- Lazy loading to avoid circular imports
- Thread-safe singleton pattern

Usage:
    from harvester_v2.agents.registry import SubAgentDiscovery

    # Discover all sub-agents
    agents = SubAgentDiscovery.discover()
    # {'scoring_subagent': ScoringSubAgent, 'nba_subagent': NBASubAgent, ...}

    # Get specific sub-agent class
    ScoringAgent = SubAgentDiscovery.get("scoring_subagent")
    agent = ScoringAgent()

    # Auto-register all discovered agents with SubAgentRegistry
    SubAgentDiscovery.register_all()
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, Optional, List
import threading

from shared.logging import get_platform_logger

logger = get_platform_logger(__name__)


class SubAgentDiscovery:
    """
    Auto-discovery for sub-agents (filesystem-based).

    Scans the agents/core/ directory for files matching *_subagent.py
    and discovers classes that inherit from BaseSubAgent.

    This complements SubAgentRegistry (parallel_executor.py) which uses
    decorator-based registration. Use register_all() to feed discovered
    agents into SubAgentRegistry.

    Thread-safe with lazy initialization.
    """

    # Directory containing sub-agent implementations
    AGENTS_DIR = Path(__file__).parent / "core"

    # Registry storage (class-level singleton)
    _registry: Dict[str, Type] = {}
    _discovered: bool = False
    _lock = threading.Lock()

    @classmethod
    def discover(cls, force: bool = False) -> Dict[str, Type]:
        """
        Scan core/ directory for sub-agent implementations.

        Discovers all Python files matching *_subagent.py and
        registers classes ending in "SubAgent" that have the
        required interface (_execute_plan method).

        Args:
            force: Re-scan even if already discovered

        Returns:
            Dict mapping module names to sub-agent classes
        """
        with cls._lock:
            if cls._discovered and not force:
                return cls._registry.copy()

            cls._registry.clear()

            if not cls.AGENTS_DIR.exists():
                logger.warning(f"Agents directory not found: {cls.AGENTS_DIR}")
                return {}

            logger.info(f"Scanning for sub-agents in: {cls.AGENTS_DIR}")

            for agent_file in cls.AGENTS_DIR.glob("*_subagent.py"):
                try:
                    cls._register_from_file(agent_file)
                except Exception as e:
                    logger.error(
                        f"Failed to load sub-agent from {agent_file.name}",
                        error=str(e)
                    )

            cls._discovered = True

            logger.info(
                f"Sub-agent discovery complete",
                count=len(cls._registry),
                agents=list(cls._registry.keys())
            )

            return cls._registry.copy()

    @classmethod
    def _register_from_file(cls, agent_file: Path) -> None:
        """
        Load and register sub-agents from a single file.

        Args:
            agent_file: Path to *_subagent.py file
        """
        module_name = f"harvester_v2.agents.core.{agent_file.stem}"

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Could not import {module_name}: {e}")
            return

        # Find classes ending in "SubAgent"
        for name, obj in vars(module).items():
            if not name.endswith("SubAgent"):
                continue

            if not inspect.isclass(obj):
                continue

            # Check for required interface
            if not hasattr(obj, "_execute_plan"):
                continue

            # Skip the base class itself
            if name == "BaseSubAgent":
                continue

            # Register using file stem as key
            cls._registry[agent_file.stem] = obj
            logger.debug(f"Registered sub-agent: {agent_file.stem} -> {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """
        Get a registered sub-agent class by name.

        Args:
            name: Sub-agent name (e.g., "scoring_subagent")

        Returns:
            Sub-agent class or None if not found
        """
        if not cls._discovered:
            cls.discover()

        return cls._registry.get(name)

    @classmethod
    def get_or_raise(cls, name: str) -> Type:
        """
        Get a registered sub-agent class, raising if not found.

        Args:
            name: Sub-agent name

        Returns:
            Sub-agent class

        Raises:
            KeyError: If sub-agent not registered
        """
        agent = cls.get(name)
        if agent is None:
            available = cls.list()
            raise KeyError(
                f"Sub-agent '{name}' not found. Available: {available}"
            )
        return agent

    @classmethod
    def list(cls) -> List[str]:
        """
        List all registered sub-agent names.

        Returns:
            List of sub-agent names
        """
        if not cls._discovered:
            cls.discover()
        return list(cls._registry.keys())

    @classmethod
    def instantiate(cls, name: str, **kwargs) -> object:
        """
        Instantiate a sub-agent by name.

        Args:
            name: Sub-agent name
            **kwargs: Arguments to pass to constructor

        Returns:
            Instantiated sub-agent
        """
        agent_cls = cls.get_or_raise(name)
        return agent_cls(**kwargs)

    @classmethod
    def instantiate_all(cls, **kwargs) -> Dict[str, object]:
        """
        Instantiate all registered sub-agents.

        Args:
            **kwargs: Arguments to pass to all constructors

        Returns:
            Dict mapping names to instances
        """
        if not cls._discovered:
            cls.discover()

        instances = {}
        for name, agent_cls in cls._registry.items():
            try:
                instances[name] = agent_cls(**kwargs)
            except Exception as e:
                logger.error(f"Failed to instantiate {name}", error=str(e))

        return instances

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        with cls._lock:
            cls._registry.clear()
            cls._discovered = False

    @classmethod
    def register_all(cls) -> int:
        """
        Register all discovered sub-agents with SubAgentRegistry.

        Integrates auto-discovery with the decorator-based SubAgentRegistry
        from parallel_executor.py for tenant-aware instantiation.

        Returns:
            Number of agents registered
        """
        from harvester_v2.agents.parallel_executor import SubAgentRegistry

        if not cls._discovered:
            cls.discover()

        count = 0
        for name, agent_cls in cls._registry.items():
            try:
                SubAgentRegistry.register(agent_cls)
                count += 1
            except Exception as e:
                logger.error(f"Failed to register {name} with SubAgentRegistry", error=str(e))

        logger.info(f"Registered {count} sub-agents with SubAgentRegistry")
        return count
