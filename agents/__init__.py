"""
Harvester v2 Sub-Agent Framework

Implements the DeepAgents-inspired pattern with:
    - Isolated execution contexts (fresh for each run)
    - Plan -> Execute -> Critique -> Output lifecycle
    - Automatic tracing and metrics via Langfuse
    - Model router integration (tenant-aware)

Core classes:
    - BaseSubAgent: Abstract base class for all sub-agents
    - SubAgentContext: Isolated execution context
    - SubAgentResult: Standardized result from sub-agent execution
    - ParallelSubAgentExecutor: Orchestrates parallel sub-agent execution
    - SubAgentRegistry: Decorator-based registration + tenant config
    - SubAgentDiscovery: Filesystem-based auto-discovery
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harvester_v2.agents.base_subagent import (
        BaseSubAgent,
        SubAgentContext,
        SubAgentResult
    )
    from harvester_v2.agents.parallel_executor import (
        ParallelSubAgentExecutor,
        ParallelExecutionResult,
        SubAgentRegistry
    )
    from harvester_v2.agents.registry import SubAgentDiscovery
