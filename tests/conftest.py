"""
Test Configuration for Harvester V2 Tests
==========================================

Provides shared fixtures and configuration for testing
the Harvester V2 intelligence engine components.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from harvester_v2.agents.base_subagent import SubAgentContext, SubAgentResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_context() -> SubAgentContext:
    """Create a sample SubAgentContext for testing."""
    return SubAgentContext(
        context_id="test-context-001",
        tenant_id="gigaclear",
        customer_id="CUST-12345",
        input_data={
            "product": "broadband_100",
            "tenure_months": 24,
            "monthly_spend": 45.00,
            "last_contact_days": 15,
            "support_tickets_30d": 2,
            "payment_issues_90d": 0,
            "usage_gb_30d": 150
        },
        memory_snapshot={
            "interaction": [
                {"content": "Customer called about slow speeds", "importance": 0.7}
            ],
            "preference": [
                {"content": "Prefers email communication", "importance": 0.6}
            ]
        },
        session_id="session-001",
        orchestrator_run_id="run-001"
    )


@pytest.fixture
def sample_context_high_risk() -> SubAgentContext:
    """Create a high-risk customer context for testing."""
    return SubAgentContext(
        context_id="test-context-002",
        tenant_id="gigaclear",
        customer_id="CUST-67890",
        input_data={
            "product": "broadband_50",
            "tenure_months": 6,
            "monthly_spend": 35.00,
            "last_contact_days": 3,
            "support_tickets_30d": 5,
            "payment_issues_90d": 2,
            "usage_gb_30d": 20,
            "mentioned_competitor": True,
            "contract_end_days": 30
        },
        memory_snapshot={
            "interaction": [
                {"content": "Customer complained about price", "importance": 0.9},
                {"content": "Asked about cancellation process", "importance": 0.95}
            ]
        }
    )


@pytest.fixture
def sample_scores() -> Dict[str, Any]:
    """Sample scoring result for testing downstream agents."""
    return {
        "fraud_score": 0.1,
        "fraud_confidence": 0.85,
        "fraud_reasoning": "Low risk - established customer",
        "fraud_factors": ["long_tenure", "no_payment_issues"],

        "churn_risk": 0.72,
        "churn_confidence": 0.80,
        "churn_reasoning": "High churn risk due to recent complaints",
        "churn_factors": ["high_support_tickets", "mentioned_competitor"],
        "churn_timeframe_days": 30,

        "propensity_score": 0.35,
        "propensity_confidence": 0.70,
        "propensity_reasoning": "Low upsell opportunity - unhappy customer",
        "propensity_opportunities": [],

        "risk_score": 0.65,
        "risk_confidence": 0.75,
        "risk_reasoning": "Elevated risk due to churn probability",
        "risk_factors": ["churn_risk", "contract_ending"],

        "customer_segment": "at_risk",
        "segment_reasoning": "High churn risk customer needing attention"
    }


@pytest.fixture
def mock_model_router():
    """Create a mock model router for testing."""
    with patch("shared.ai.model_router.get_model_router") as mock:
        router = MagicMock()
        router.complete = AsyncMock(return_value='{"test": "response"}')
        mock.return_value = router
        yield router


@pytest.fixture
def mock_weaviate_client():
    """Create a mock Weaviate client for testing."""
    with patch("shared.database.weaviate_client.create_weaviate_client") as mock:
        client = MagicMock()
        client.is_ready.return_value = True
        client.schema.get.return_value = {"classes": []}
        client.query.get.return_value = MagicMock(
            with_near_text=MagicMock(return_value=MagicMock(
                with_where=MagicMock(return_value=MagicMock(
                    with_limit=MagicMock(return_value=MagicMock(
                        do=MagicMock(return_value={
                            "data": {"Get": {"EpisodicMemory": []}}
                        })
                    ))
                ))
            ))
        )
        mock.return_value = client
        yield client


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Test Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (requires services)"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests (full stack)"
    )
