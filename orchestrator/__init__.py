"""
Harvester v2 Orchestrator Layer

Contains the main intelligence orchestrator that coordinates all sub-agents,
manages execution context, and integrates with LangGraph for workflow orchestration.

Components:
    - IntelligenceOrchestrator: Main entry point for customer intelligence analysis
    - HarvesterV2Graph: LangGraph-based workflow with checkpointing
    - ExecutionContext: State tracking for orchestrator runs
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harvester_v2.orchestrator.intelligence_orchestrator import (
        IntelligenceOrchestrator,
        IntelligenceRequest,
        IntelligenceResult
    )
    from harvester_v2.orchestrator.intelligence_graph import HarvesterV2Graph, HarvesterV2State
