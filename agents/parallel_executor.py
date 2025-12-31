"""
Parallel Sub-Agent Executor for Harvester V2
=============================================

Executes multiple sub-agents concurrently using asyncio.gather().
Each sub-agent runs with an isolated context to prevent interference.

Key features:
- Parallel execution for independent sub-agents
- Phased execution for dependent sub-agents
- Automatic result aggregation
- Error isolation (one failure doesn't stop others)
- Context isolation (each agent gets a copy)

Usage:
    # Create executor with sub-agents
    executor = ParallelSubAgentExecutor([
        ScoringSubAgent(),
        CausalitySubAgent(),
        NBASubAgent()
    ])

    # Execute in parallel
    result = await executor.execute_parallel(
        base_context=context,
        subagent_names=["scoring_subagent", "causality_subagent"],
        task_description="Analyze customer risk"
    )

    # Or execute in phases (dependent results)
    result = await executor.execute_phased(
        base_context=context,
        phases=[
            ["scoring_subagent"],           # Phase 1: Independent
            ["causality_subagent"],         # Phase 2: Uses scoring results
            ["nba_subagent"]                # Phase 3: Uses both
        ],
        task_description="Full customer analysis"
    )
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import structlog

from harvester_v2.agents.base_subagent import (
    BaseSubAgent,
    SubAgentContext,
    SubAgentResult
)
from shared.logging import get_platform_logger

logger = get_platform_logger(__name__)


@dataclass
class ParallelExecutionResult:
    """
    Result of parallel sub-agent execution.

    Aggregates results from multiple sub-agents with metadata.

    Attributes:
        success: True if ALL sub-agents succeeded
        partial_success: True if at least one sub-agent succeeded
        results: Dict mapping agent name to SubAgentResult
        merged_data: Merged output data from all successful agents
        total_execution_time_ms: Total wall-clock time
        failed_agents: List of agents that failed
        phase_times_ms: Execution times per phase (for phased execution)
    """
    success: bool
    partial_success: bool
    results: Dict[str, SubAgentResult]
    merged_data: Dict[str, Any]
    total_execution_time_ms: int
    failed_agents: List[str] = field(default_factory=list)
    phase_times_ms: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_result(self, agent_name: str) -> Optional[SubAgentResult]:
        """Get result for a specific agent."""
        return self.results.get(agent_name)

    def get_data(self, agent_name: str) -> Dict[str, Any]:
        """Get output data for a specific agent."""
        result = self.results.get(agent_name)
        return result.data if result and result.success else {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "partial_success": self.partial_success,
            "results": {
                name: result.to_dict()
                for name, result in self.results.items()
            },
            "merged_data": self.merged_data,
            "total_execution_time_ms": self.total_execution_time_ms,
            "failed_agents": self.failed_agents,
            "phase_times_ms": self.phase_times_ms,
            "metadata": self.metadata
        }


class ParallelSubAgentExecutor:
    """
    Executes sub-agents in parallel with isolated contexts.

    This executor manages concurrent execution of multiple sub-agents,
    ensuring each gets an isolated context and aggregating results.

    Execution modes:
    1. execute_parallel: Run specified agents concurrently
    2. execute_phased: Run in phases where later phases get earlier results
    3. execute_all: Run all registered agents in parallel

    Error handling:
    - Failures are isolated: one agent failing doesn't stop others
    - Results include both successful and failed agents
    - Partial success is tracked separately from full success
    """

    def __init__(
        self,
        subagents: List[BaseSubAgent],
        max_concurrency: int = 10
    ):
        """
        Initialize executor with sub-agents.

        Args:
            subagents: List of sub-agent instances
            max_concurrency: Maximum concurrent executions (default 10)
        """
        self.subagents: Dict[str, BaseSubAgent] = {
            sa.agent_name: sa for sa in subagents
        }
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

        logger.info(
            "ParallelSubAgentExecutor initialized",
            agents=list(self.subagents.keys()),
            max_concurrency=max_concurrency
        )

    def register_subagent(self, subagent: BaseSubAgent) -> None:
        """
        Register a new sub-agent.

        Args:
            subagent: Sub-agent instance to register
        """
        self.subagents[subagent.agent_name] = subagent
        logger.info("Sub-agent registered", agent=subagent.agent_name)

    def unregister_subagent(self, agent_name: str) -> bool:
        """
        Unregister a sub-agent.

        Args:
            agent_name: Name of agent to unregister

        Returns:
            True if agent was found and removed
        """
        if agent_name in self.subagents:
            del self.subagents[agent_name]
            logger.info("Sub-agent unregistered", agent=agent_name)
            return True
        return False

    async def execute_parallel(
        self,
        base_context: SubAgentContext,
        subagent_names: List[str],
        task_description: str
    ) -> ParallelExecutionResult:
        """
        Execute specified sub-agents in parallel.

        Each sub-agent gets a COPY of the context (isolated).
        All agents run concurrently using asyncio.gather().

        Args:
            base_context: Base context to copy for each agent
            subagent_names: Names of agents to execute
            task_description: Task description for all agents

        Returns:
            ParallelExecutionResult with aggregated results
        """
        start_time = datetime.utcnow()

        # Filter to requested sub-agents
        agents_to_run = [
            self.subagents[name] for name in subagent_names
            if name in self.subagents
        ]

        missing_agents = set(subagent_names) - set(self.subagents.keys())
        if missing_agents:
            logger.warning(
                "Some requested agents not found",
                missing=list(missing_agents)
            )

        if not agents_to_run:
            return ParallelExecutionResult(
                success=False,
                partial_success=False,
                results={},
                merged_data={},
                total_execution_time_ms=0,
                failed_agents=list(missing_agents)
            )

        # Create tasks with isolated contexts
        async def run_with_semaphore(agent: BaseSubAgent) -> SubAgentResult:
            """Run agent with semaphore for concurrency control."""
            async with self._semaphore:
                # CRITICAL: Each agent gets its own context copy
                isolated_context = base_context.copy()
                return await agent.run_isolated(isolated_context, task_description)

        # Execute all in parallel
        logger.info(
            "Executing sub-agents in parallel",
            agents=[a.agent_name for a in agents_to_run],
            task=task_description[:50]
        )

        results = await asyncio.gather(
            *[run_with_semaphore(agent) for agent in agents_to_run],
            return_exceptions=True
        )

        # Process results
        result_dict: Dict[str, SubAgentResult] = {}
        merged_data: Dict[str, Any] = {}
        failed_agents: List[str] = []

        for agent, result in zip(agents_to_run, results):
            if isinstance(result, Exception):
                # Agent raised an exception
                logger.error(
                    "Sub-agent raised exception",
                    subagent=agent.agent_name,
                    error=str(result)
                )
                failed_agents.append(agent.agent_name)
                result_dict[agent.agent_name] = SubAgentResult(
                    subagent_name=agent.agent_name,
                    success=False,
                    data={},
                    reasoning_trace="",
                    critique_score=0.0,
                    execution_time_ms=0,
                    model_used="",
                    error=str(result)
                )
            elif result.success:
                result_dict[agent.agent_name] = result
                # Merge successful data
                merged_data[agent.agent_name] = result.data
            else:
                result_dict[agent.agent_name] = result
                failed_agents.append(agent.agent_name)

        # Calculate execution time
        execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        # Build result
        success = len(failed_agents) == 0
        partial_success = len(failed_agents) < len(agents_to_run)

        logger.info(
            "Parallel execution completed",
            success=success,
            partial_success=partial_success,
            successful_agents=len(agents_to_run) - len(failed_agents),
            failed_agents=failed_agents,
            execution_time_ms=execution_time
        )

        return ParallelExecutionResult(
            success=success,
            partial_success=partial_success,
            results=result_dict,
            merged_data=merged_data,
            total_execution_time_ms=execution_time,
            failed_agents=failed_agents
        )

    async def execute_phased(
        self,
        base_context: SubAgentContext,
        phases: List[List[str]],
        task_description: str
    ) -> ParallelExecutionResult:
        """
        Execute sub-agents in phases.

        Phase N+1 can use results from Phase N.
        Sub-agents within a phase run in parallel.

        Example phases:
            [
                ["scoring_subagent"],           # Phase 1
                ["causality_subagent"],         # Phase 2: Gets scoring results
                ["nba_subagent"]                # Phase 3: Gets both results
            ]

        Args:
            base_context: Base context to start with
            phases: List of phase groups (each group runs in parallel)
            task_description: Task description for all agents

        Returns:
            ParallelExecutionResult with aggregated results from all phases
        """
        start_time = datetime.utcnow()

        current_context = base_context
        all_results: Dict[str, SubAgentResult] = {}
        all_merged: Dict[str, Any] = {}
        all_failed: List[str] = []
        phase_times: List[int] = []

        for phase_idx, phase_agents in enumerate(phases):
            phase_start = datetime.utcnow()

            logger.info(
                f"Executing phase {phase_idx + 1}/{len(phases)}",
                agents=phase_agents
            )

            # Execute this phase in parallel
            result = await self.execute_parallel(
                current_context,
                phase_agents,
                task_description
            )

            # Record phase time
            phase_time = int((datetime.utcnow() - phase_start).total_seconds() * 1000)
            phase_times.append(phase_time)

            # Accumulate results
            all_results.update(result.results)
            all_merged.update(result.merged_data)
            all_failed.extend(result.failed_agents)

            # Update context for next phase with merged data
            # This allows later phases to access earlier results
            current_context = SubAgentContext(
                context_id=base_context.context_id,
                tenant_id=base_context.tenant_id,
                customer_id=base_context.customer_id,
                input_data={**base_context.input_data, **all_merged},
                memory_snapshot=base_context.memory_snapshot,
                session_id=base_context.session_id,
                orchestrator_run_id=base_context.orchestrator_run_id,
                langfuse_trace_id=base_context.langfuse_trace_id
            )

            logger.info(
                f"Phase {phase_idx + 1} completed",
                phase_time_ms=phase_time,
                successful=len(phase_agents) - len([a for a in phase_agents if a in result.failed_agents])
            )

        # Calculate total execution time
        total_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        success = len(all_failed) == 0
        total_agents = sum(len(phase) for phase in phases)
        partial_success = len(all_failed) < total_agents

        logger.info(
            "Phased execution completed",
            phases=len(phases),
            success=success,
            partial_success=partial_success,
            total_time_ms=total_time,
            phase_times_ms=phase_times
        )

        return ParallelExecutionResult(
            success=success,
            partial_success=partial_success,
            results=all_results,
            merged_data=all_merged,
            total_execution_time_ms=total_time,
            failed_agents=all_failed,
            phase_times_ms=phase_times,
            metadata={"phases": len(phases)}
        )

    async def execute_all(
        self,
        base_context: SubAgentContext,
        task_description: str
    ) -> ParallelExecutionResult:
        """
        Execute all registered sub-agents in parallel.

        Convenience method that runs all registered agents.

        Args:
            base_context: Base context to copy for each agent
            task_description: Task description for all agents

        Returns:
            ParallelExecutionResult with all results
        """
        return await self.execute_parallel(
            base_context,
            list(self.subagents.keys()),
            task_description
        )

    def get_agent_names(self) -> List[str]:
        """Get list of all registered agent names."""
        return list(self.subagents.keys())

    def get_agent(self, name: str) -> Optional[BaseSubAgent]:
        """Get a specific agent by name."""
        return self.subagents.get(name)


class SubAgentRegistry:
    """
    Registry for managing sub-agent instances per tenant.

    Provides factory methods for creating tenant-specific sub-agent sets.
    This allows different tenants to have different agent configurations.

    Usage:
        # Get sub-agents for a tenant
        subagents = SubAgentRegistry.get_subagents_for_tenant("gigaclear")

        # Create executor with tenant sub-agents
        executor = ParallelSubAgentExecutor(subagents)
    """

    # Registry of available sub-agent classes
    _registry: Dict[str, type] = {}

    # Per-tenant configuration overrides
    _tenant_configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, agent_class: type) -> type:
        """
        Register a sub-agent class.

        Can be used as a decorator:
            @SubAgentRegistry.register
            class MyScoringSubAgent(BaseSubAgent):
                ...

        Args:
            agent_class: Sub-agent class to register

        Returns:
            The same class (for decorator use)
        """
        # Get agent name from class or instance
        if hasattr(agent_class, 'agent_name'):
            name = agent_class.agent_name
        else:
            # Convert class name to snake_case
            name = cls._to_snake_case(agent_class.__name__)

        cls._registry[name] = agent_class
        logger.info("Registered sub-agent class", name=name, cls=agent_class.__name__)
        return agent_class

    @classmethod
    def set_tenant_config(
        cls,
        tenant_id: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Set configuration overrides for a tenant.

        Args:
            tenant_id: Tenant identifier
            config: Configuration dict (agent names -> options)
        """
        cls._tenant_configs[tenant_id] = config
        logger.info("Set tenant config", tenant_id=tenant_id, agents=list(config.keys()))

    @classmethod
    def get_subagents_for_tenant(
        cls,
        tenant_id: str,
        agent_names: Optional[List[str]] = None
    ) -> List[BaseSubAgent]:
        """
        Get sub-agent instances configured for a tenant.

        Args:
            tenant_id: Tenant identifier
            agent_names: Specific agents to include (None = all)

        Returns:
            List of configured sub-agent instances
        """
        tenant_config = cls._tenant_configs.get(tenant_id, {})
        subagents = []

        names_to_create = agent_names or list(cls._registry.keys())

        for name in names_to_create:
            if name not in cls._registry:
                logger.warning("Sub-agent not registered", name=name)
                continue

            agent_class = cls._registry[name]
            agent_config = tenant_config.get(name, {})

            # Create instance with tenant-specific config
            try:
                subagent = agent_class(options=agent_config)
                subagents.append(subagent)
            except Exception as e:
                logger.error(
                    "Failed to create sub-agent",
                    name=name,
                    error=str(e)
                )

        logger.info(
            "Created sub-agents for tenant",
            tenant_id=tenant_id,
            agents=[sa.agent_name for sa in subagents]
        )

        return subagents

    @classmethod
    def get_registered_agents(cls) -> List[str]:
        """Get list of all registered agent names."""
        return list(cls._registry.keys())

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
