"""
Harvester v2 - Claude-Native Intelligence Engine

A complete rebuild of the Harvester system following the DeepAgents-inspired
sub-agent pattern, integrated with LangGraph orchestration and Langfuse observability.

Architecture:
    - Single orchestrator agent with todo-style planning
    - Parallel sub-agents with isolated contexts (scoring, causality, NBA, retention)
    - MCP tool layer for external data access
    - Memory system (episodic, semantic, procedural)
    - Learning pipeline for continuous improvement

Key integrations:
    - LangGraph StateGraph for workflow orchestration and checkpointing
    - Langfuse for end-to-end observability and tracing
    - AsyncPostgresSaver for durable checkpoint persistence
    - AmbientGraphRuntime for graceful degradation
"""

__version__ = "2.0.0"
__author__ = "nexgAI"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harvester_v2.orchestrator.intelligence_orchestrator import IntelligenceOrchestrator
    from harvester_v2.orchestrator.intelligence_graph import HarvesterV2Graph
    from harvester_v2.agents.base_subagent import BaseSubAgent, SubAgentContext, SubAgentResult
    from harvester_v2.agents.parallel_executor import ParallelSubAgentExecutor
