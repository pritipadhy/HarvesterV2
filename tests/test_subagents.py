"""
Sub-Agent Tests for Harvester V2
=================================

Tests for the core sub-agents:
- ScoringSubAgent
- CausalitySubAgent
- NBASubAgent
- ReasoningSubAgent
- RetainerStrategySubAgent
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from harvester_v2.agents.base_subagent import (
    BaseSubAgent,
    SubAgentContext,
    SubAgentResult,
    SubAgentStage
)
from harvester_v2.agents.core.scoring_subagent import ScoringSubAgent
from harvester_v2.agents.core.causality_subagent import CausalitySubAgent
from harvester_v2.agents.core.nba_subagent import NBASubAgent
from harvester_v2.agents.core.reasoning_subagent import ReasoningSubAgent
from harvester_v2.agents.core.retainer_strategy_subagent import RetainerStrategySubAgent


# ============================================================================
# SubAgentContext Tests
# ============================================================================

class TestSubAgentContext:
    """Tests for SubAgentContext."""

    def test_context_creation(self):
        """Test context creation with defaults."""
        context = SubAgentContext(tenant_id="test")

        assert context.context_id is not None
        assert context.tenant_id == "test"
        assert context.customer_id is None
        assert context.input_data == {}
        assert context.memory_snapshot == {}

    def test_context_with_data(self, sample_context):
        """Test context with populated data."""
        assert sample_context.tenant_id == "gigaclear"
        assert sample_context.customer_id == "CUST-12345"
        assert "product" in sample_context.input_data
        assert "interaction" in sample_context.memory_snapshot

    def test_context_copy(self, sample_context):
        """Test context copy creates isolated instance."""
        copy = sample_context.copy()

        assert copy.context_id != sample_context.context_id
        assert copy.tenant_id == sample_context.tenant_id
        assert copy.input_data == sample_context.input_data

        # Modify copy shouldn't affect original
        copy.input_data["new_field"] = "value"
        assert "new_field" not in sample_context.input_data

    def test_to_prompt_context(self, sample_context):
        """Test conversion to prompt format."""
        prompt = sample_context.to_prompt_context()

        assert "gigaclear" in prompt
        assert "CUST-12345" in prompt
        assert "INPUT DATA" in prompt
        assert "RELEVANT MEMORY" in prompt


# ============================================================================
# SubAgentResult Tests
# ============================================================================

class TestSubAgentResult:
    """Tests for SubAgentResult."""

    def test_result_creation(self):
        """Test result creation."""
        result = SubAgentResult(
            subagent_name="test_agent",
            success=True,
            data={"score": 0.85},
            reasoning_trace="Test reasoning",
            critique_score=0.90,
            execution_time_ms=150,
            model_used="claude-sonnet"
        )

        assert result.subagent_name == "test_agent"
        assert result.success is True
        assert result.data["score"] == 0.85
        assert result.critique_score == 0.90

    def test_result_to_dict(self):
        """Test result serialization."""
        result = SubAgentResult(
            subagent_name="test",
            success=True,
            data={"key": "value"},
            reasoning_trace="trace",
            critique_score=0.85,
            execution_time_ms=100,
            model_used="test"
        )

        d = result.to_dict()
        assert d["subagent_name"] == "test"
        assert d["success"] is True
        assert "data" in d


# ============================================================================
# ScoringSubAgent Tests
# ============================================================================

class TestScoringSubAgent:
    """Tests for ScoringSubAgent."""

    @pytest.fixture
    def scoring_agent(self):
        """Create scoring agent with mocked router."""
        with patch("harvester_v2.agents.base_subagent.get_model_router") as mock:
            router = MagicMock()
            router.complete = AsyncMock()
            mock.return_value = router
            agent = ScoringSubAgent()
            agent.model_router = router
            yield agent

    @pytest.mark.asyncio
    async def test_plan_generation(self, scoring_agent, sample_context):
        """Test scoring plan generation."""
        scoring_agent.model_router.complete.return_value = json.dumps({
            "scores_to_calculate": ["fraud", "churn", "propensity", "risk"],
            "fraud_signals": ["payment_issues"],
            "churn_signals": ["support_tickets"],
            "propensity_signals": ["usage"],
            "risk_signals": ["tenure"],
            "data_quality_notes": "Good data quality"
        })

        plan = await scoring_agent._plan(sample_context, "Calculate risk scores")

        assert "scores_to_calculate" in plan
        assert "fraud" in plan["scores_to_calculate"]

    @pytest.mark.asyncio
    async def test_execute_plan(self, scoring_agent, sample_context):
        """Test scoring execution."""
        scoring_agent.model_router.complete.return_value = json.dumps({
            "fraud_score": 0.15,
            "fraud_confidence": 0.85,
            "fraud_reasoning": "Low risk",
            "fraud_factors": ["good_history"],
            "churn_risk": 0.45,
            "churn_confidence": 0.80,
            "churn_reasoning": "Moderate risk",
            "churn_factors": ["support_tickets"],
            "churn_timeframe_days": 60,
            "propensity_score": 0.60,
            "propensity_confidence": 0.75,
            "propensity_reasoning": "Good upsell potential",
            "propensity_opportunities": ["upgrade"],
            "risk_score": 0.35,
            "risk_confidence": 0.80,
            "risk_reasoning": "Moderate overall risk",
            "risk_factors": ["tenure"],
            "customer_segment": "growth",
            "segment_reasoning": "Growing customer",
            "scoring_summary": "Healthy customer with upsell potential"
        })

        plan = {"scores_to_calculate": ["fraud", "churn", "propensity", "risk"]}
        result = await scoring_agent._execute_plan(sample_context, plan)

        assert "fraud_score" in result
        assert 0 <= result["fraud_score"] <= 1
        assert "churn_risk" in result
        assert "customer_id" in result

    @pytest.mark.asyncio
    async def test_run_isolated(self, scoring_agent, sample_context):
        """Test full isolated execution."""
        # Mock all LLM calls
        scoring_agent.model_router.complete.side_effect = [
            # Plan response
            json.dumps({
                "scores_to_calculate": ["fraud", "churn"],
                "fraud_signals": [],
                "churn_signals": [],
                "propensity_signals": [],
                "risk_signals": [],
                "data_quality_notes": ""
            }),
            # Execute response
            json.dumps({
                "fraud_score": 0.2,
                "fraud_confidence": 0.8,
                "fraud_reasoning": "Low risk",
                "fraud_factors": [],
                "churn_risk": 0.3,
                "churn_confidence": 0.8,
                "churn_reasoning": "Low risk",
                "churn_factors": [],
                "churn_timeframe_days": 90,
                "propensity_score": 0.5,
                "propensity_confidence": 0.7,
                "propensity_reasoning": "Moderate",
                "propensity_opportunities": [],
                "risk_score": 0.25,
                "risk_confidence": 0.8,
                "risk_reasoning": "Low risk",
                "risk_factors": [],
                "customer_segment": "nurture",
                "segment_reasoning": "New customer",
                "scoring_summary": "Healthy customer"
            }),
            # Critique response
            json.dumps({
                "completeness": 0.9,
                "accuracy": 0.85,
                "actionability": 0.9,
                "overall_score": 0.88,
                "reasoning": "Good quality scores"
            })
        ]

        result = await scoring_agent.run_isolated(
            sample_context,
            "Calculate customer risk scores"
        )

        assert result.success is True
        assert result.subagent_name == "scoring_subagent"
        assert result.critique_score >= 0.85
        assert "fraud_score" in result.data

    def test_normalize_scores(self, scoring_agent):
        """Test score normalization."""
        scores = {
            "fraud_score": 1.5,  # Should be clamped to 1.0
            "churn_risk": -0.2,  # Should be clamped to 0.0
            "propensity_score": 0.75  # Should stay as is
        }

        normalized = scoring_agent._normalize_scores(scores)

        assert normalized["fraud_score"] == 1.0
        assert normalized["churn_risk"] == 0.0
        assert normalized["propensity_score"] == 0.75

    def test_get_alerts(self, scoring_agent):
        """Test alert generation from scores."""
        high_risk_scores = {
            "fraud_score": 0.8,
            "fraud_reasoning": "Suspicious activity",
            "fraud_factors": ["unusual_pattern"],
            "churn_risk": 0.75,
            "churn_reasoning": "High churn probability",
            "churn_factors": ["complaints"],
            "churn_timeframe_days": 14,
            "propensity_score": 0.7,
            "propensity_reasoning": "Good opportunity",
            "propensity_opportunities": ["upgrade"]
        }

        alerts = scoring_agent.get_alerts(high_risk_scores)

        assert len(alerts) >= 2  # Fraud and churn alerts
        fraud_alert = next(a for a in alerts if a["type"] == "fraud_alert")
        assert fraud_alert["severity"] == "high"


# ============================================================================
# CausalitySubAgent Tests
# ============================================================================

class TestCausalitySubAgent:
    """Tests for CausalitySubAgent."""

    @pytest.fixture
    def causality_agent(self):
        """Create causality agent with mocked router."""
        with patch("harvester_v2.agents.base_subagent.get_model_router") as mock:
            router = MagicMock()
            router.complete = AsyncMock()
            mock.return_value = router
            agent = CausalitySubAgent()
            agent.model_router = router
            yield agent

    @pytest.mark.asyncio
    async def test_plan_identifies_outcome(self, causality_agent, sample_context):
        """Test plan identifies target outcome."""
        sample_context.input_data["target_outcome"] = "churn"

        plan = await causality_agent._plan(sample_context, "Analyze causes")

        assert plan["target_outcome"] == "churn"
        assert "root_cause" in plan["analysis_types"]


# ============================================================================
# NBASubAgent Tests
# ============================================================================

class TestNBASubAgent:
    """Tests for NBASubAgent."""

    @pytest.fixture
    def nba_agent(self):
        """Create NBA agent with mocked router."""
        with patch("harvester_v2.agents.base_subagent.get_model_router") as mock:
            router = MagicMock()
            router.complete = AsyncMock()
            mock.return_value = router
            agent = NBASubAgent()
            agent.model_router = router
            yield agent

    @pytest.mark.asyncio
    async def test_plan_recommends_retention(self, nba_agent, sample_context):
        """Test plan recommends retention for high churn."""
        sample_context.input_data["scores"] = {"churn_risk": 0.8}

        plan = await nba_agent._plan(sample_context, "Generate NBA")

        assert plan["should_recommend_retention"] is True


# ============================================================================
# ReasoningSubAgent Tests
# ============================================================================

class TestReasoningSubAgent:
    """Tests for ReasoningSubAgent."""

    def test_higher_critique_threshold(self):
        """Test reasoning agent has higher quality bar."""
        with patch("harvester_v2.agents.base_subagent.get_model_router"):
            agent = ReasoningSubAgent()
            assert agent.critique_threshold == 0.90


# ============================================================================
# RetainerStrategySubAgent Tests
# ============================================================================

class TestRetainerStrategySubAgent:
    """Tests for RetainerStrategySubAgent."""

    @pytest.fixture
    def retention_agent(self):
        """Create retention agent with mocked router."""
        with patch("harvester_v2.agents.base_subagent.get_model_router") as mock:
            router = MagicMock()
            router.complete = AsyncMock()
            mock.return_value = router
            agent = RetainerStrategySubAgent()
            agent.model_router = router
            yield agent

    @pytest.mark.asyncio
    async def test_plan_urgency_levels(self, retention_agent, sample_context):
        """Test urgency level calculation."""
        sample_context.input_data["scores"] = {"churn_risk": 0.9}

        plan = await retention_agent._plan(sample_context, "Generate retention")

        assert plan["urgency_level"] == "critical"
