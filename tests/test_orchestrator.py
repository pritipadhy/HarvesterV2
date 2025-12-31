"""
Orchestrator Tests for Harvester V2
====================================

Tests for the main intelligence orchestrator and LangGraph integration.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from harvester_v2.agents.base_subagent import SubAgentContext, SubAgentResult
from harvester_v2.agents.parallel_executor import (
    ParallelSubAgentExecutor,
    ParallelExecutionResult,
    SubAgentRegistry
)


# ============================================================================
# ParallelSubAgentExecutor Tests
# ============================================================================

class TestParallelSubAgentExecutor:
    """Tests for ParallelSubAgentExecutor."""

    @pytest.fixture
    def mock_subagents(self):
        """Create mock sub-agents for testing."""
        agents = []
        for name in ["agent_a", "agent_b", "agent_c"]:
            agent = MagicMock()
            agent.agent_name = name
            agent.run_isolated = AsyncMock(return_value=SubAgentResult(
                subagent_name=name,
                success=True,
                data={"result": f"data_from_{name}"},
                reasoning_trace="Test reasoning",
                critique_score=0.90,
                execution_time_ms=100,
                model_used="test"
            ))
            agents.append(agent)
        return agents

    @pytest.fixture
    def executor(self, mock_subagents):
        """Create executor with mock agents."""
        return ParallelSubAgentExecutor(mock_subagents)

    @pytest.mark.asyncio
    async def test_execute_parallel(self, executor, sample_context):
        """Test parallel execution of sub-agents."""
        result = await executor.execute_parallel(
            base_context=sample_context,
            subagent_names=["agent_a", "agent_b"],
            task_description="Test task"
        )

        assert result.success is True
        assert len(result.results) == 2
        assert "agent_a" in result.results
        assert "agent_b" in result.results
        assert result.merged_data["agent_a"]["result"] == "data_from_agent_a"

    @pytest.mark.asyncio
    async def test_execute_parallel_with_failure(self, executor, sample_context):
        """Test parallel execution with one failing agent."""
        # Make agent_b fail
        executor.subagents["agent_b"].run_isolated = AsyncMock(
            return_value=SubAgentResult(
                subagent_name="agent_b",
                success=False,
                data={},
                reasoning_trace="",
                critique_score=0.0,
                execution_time_ms=50,
                model_used="",
                error="Test error"
            )
        )

        result = await executor.execute_parallel(
            base_context=sample_context,
            subagent_names=["agent_a", "agent_b"],
            task_description="Test task"
        )

        assert result.success is False
        assert result.partial_success is True
        assert "agent_b" in result.failed_agents
        assert "agent_a" in result.merged_data  # Successful agent data still included

    @pytest.mark.asyncio
    async def test_execute_phased(self, executor, sample_context):
        """Test phased execution."""
        phases = [
            ["agent_a"],           # Phase 1
            ["agent_b", "agent_c"]  # Phase 2
        ]

        result = await executor.execute_phased(
            base_context=sample_context,
            phases=phases,
            task_description="Test task"
        )

        assert result.success is True
        assert len(result.results) == 3
        assert len(result.phase_times_ms) == 2  # Two phases

    @pytest.mark.asyncio
    async def test_context_isolation(self, executor, sample_context):
        """Test that each agent gets isolated context."""
        contexts_seen = []

        async def capture_context(context, task):
            contexts_seen.append(context.context_id)
            return SubAgentResult(
                subagent_name="test",
                success=True,
                data={},
                reasoning_trace="",
                critique_score=0.9,
                execution_time_ms=10,
                model_used=""
            )

        for agent in executor.subagents.values():
            agent.run_isolated = capture_context

        await executor.execute_parallel(
            base_context=sample_context,
            subagent_names=["agent_a", "agent_b"],
            task_description="Test"
        )

        # Each agent should get a different context ID (isolated)
        assert len(set(contexts_seen)) == 2

    def test_register_subagent(self, executor):
        """Test agent registration."""
        new_agent = MagicMock()
        new_agent.agent_name = "agent_d"

        executor.register_subagent(new_agent)

        assert "agent_d" in executor.subagents

    def test_unregister_subagent(self, executor):
        """Test agent unregistration."""
        result = executor.unregister_subagent("agent_a")

        assert result is True
        assert "agent_a" not in executor.subagents

    def test_get_agent_names(self, executor):
        """Test getting agent names."""
        names = executor.get_agent_names()

        assert len(names) == 3
        assert "agent_a" in names


# ============================================================================
# SubAgentRegistry Tests
# ============================================================================

class TestSubAgentRegistry:
    """Tests for SubAgentRegistry."""

    def test_register_class(self):
        """Test class registration."""
        # Create a mock class
        class MockSubAgent:
            agent_name = "mock_agent"

            def __init__(self, options=None):
                self.options = options

        # Clear any existing registrations
        SubAgentRegistry._registry.clear()

        # Register
        SubAgentRegistry.register(MockSubAgent)

        assert "mock_agent" in SubAgentRegistry._registry

    def test_set_tenant_config(self):
        """Test tenant configuration."""
        config = {
            "scoring_subagent": {"critique_threshold": 0.90}
        }

        SubAgentRegistry.set_tenant_config("test_tenant", config)

        assert "test_tenant" in SubAgentRegistry._tenant_configs

    def test_to_snake_case(self):
        """Test snake_case conversion."""
        assert SubAgentRegistry._to_snake_case("CamelCase") == "camel_case"
        # Acronyms are kept together (sensible behavior)
        assert SubAgentRegistry._to_snake_case("HTTPServer") == "http_server"
        assert SubAgentRegistry._to_snake_case("MyAgent") == "my_agent"
        assert SubAgentRegistry._to_snake_case("ScoringSubAgent") == "scoring_sub_agent"


# ============================================================================
# ParallelExecutionResult Tests
# ============================================================================

class TestParallelExecutionResult:
    """Tests for ParallelExecutionResult."""

    def test_result_creation(self):
        """Test result creation."""
        result = ParallelExecutionResult(
            success=True,
            partial_success=True,
            results={"agent_a": MagicMock()},
            merged_data={"agent_a": {"key": "value"}},
            total_execution_time_ms=200
        )

        assert result.success is True
        assert result.total_execution_time_ms == 200

    def test_get_data(self):
        """Test getting agent data."""
        mock_result = SubAgentResult(
            subagent_name="test",
            success=True,
            data={"score": 0.85},
            reasoning_trace="",
            critique_score=0.9,
            execution_time_ms=100,
            model_used=""
        )

        result = ParallelExecutionResult(
            success=True,
            partial_success=True,
            results={"test": mock_result},
            merged_data={"test": {"score": 0.85}},
            total_execution_time_ms=100
        )

        data = result.get_data("test")
        assert data["score"] == 0.85

    def test_to_dict(self):
        """Test serialization."""
        mock_result = SubAgentResult(
            subagent_name="test",
            success=True,
            data={},
            reasoning_trace="",
            critique_score=0.9,
            execution_time_ms=100,
            model_used=""
        )

        result = ParallelExecutionResult(
            success=True,
            partial_success=True,
            results={"test": mock_result},
            merged_data={},
            total_execution_time_ms=100
        )

        d = result.to_dict()
        assert "success" in d
        assert "results" in d
        assert "test" in d["results"]
