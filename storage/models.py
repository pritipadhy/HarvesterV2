"""
HarvesterV2 Database Models
===========================

PostgreSQL models for customer intelligence data storage.

Tables:
    - harvester_customers: Customer profiles and context
    - harvester_signals: Customer signals/events (time-series)
    - harvester_results: Intelligence pipeline outputs
    - harvester_actions: Recommended actions tracking

Usage:
    from harvester_v2.storage.models import CustomerProfile, CustomerSignal

    # Create customer
    customer = CustomerProfile(
        customer_id="CUST-001",
        tenant_id="gigaclear",
        context={"industry": "telecom", ...}
    )
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, JSON, Text,
    ForeignKey, Index, Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class SignalType(enum.Enum):
    """Types of customer signals."""
    CHURN = "churn"
    FRAUD = "fraud"
    UPSELL = "upsell"
    SUPPORT = "support"
    BILLING = "billing"
    USAGE = "usage"
    FEEDBACK = "feedback"
    COMPETITOR = "competitor"
    CONTRACT = "contract"
    INTERACTION = "interaction"


class SignalSeverity(enum.Enum):
    """Signal severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionStatus(enum.Enum):
    """Status of recommended actions."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class CustomerProfile(Base):
    """
    Customer profile with context data.

    Stores the customer's current state and all contextual information
    needed for intelligence analysis.
    """
    __tablename__ = "harvester_customers"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(100), nullable=False, index=True)
    tenant_id = Column(String(50), nullable=False, index=True)

    # Customer profile
    name = Column(String(255))
    email = Column(String(255))
    phone = Column(String(50))
    segment = Column(String(50))  # premium, standard, basic
    industry = Column(String(100))

    # Product information
    product = Column(String(100))
    monthly_revenue = Column(Float)
    tenure_months = Column(Integer)
    contract_start_date = Column(DateTime)
    contract_end_date = Column(DateTime)

    # Risk scores (cached from last analysis)
    churn_risk = Column(Float, default=0.0)
    fraud_risk = Column(Float, default=0.0)
    propensity_score = Column(Float, default=0.0)
    nps_score = Column(Integer)

    # Full context as JSONB for flexibility
    context = Column(JSONB, default={})

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_analyzed_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

    # Relationships
    signals = relationship("CustomerSignal", back_populates="customer", lazy="dynamic")
    results = relationship("IntelligenceResult", back_populates="customer", lazy="dynamic")
    actions = relationship("RecommendedAction", back_populates="customer", lazy="dynamic")

    # Indexes
    __table_args__ = (
        Index("ix_harvester_customers_tenant_customer", "tenant_id", "customer_id", unique=True),
        Index("ix_harvester_customers_churn_risk", "tenant_id", "churn_risk"),
        Index("ix_harvester_customers_segment", "tenant_id", "segment"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pipeline input."""
        return {
            "customer_id": self.customer_id,
            "tenant_id": self.tenant_id,
            "context": {
                "name": self.name,
                "email": self.email,
                "segment": self.segment,
                "industry": self.industry,
                "product": self.product,
                "monthly_revenue": self.monthly_revenue,
                "tenure_months": self.tenure_months,
                "contract_end_date": self.contract_end_date.isoformat() if self.contract_end_date else None,
                "churn_risk": self.churn_risk,
                "fraud_risk": self.fraud_risk,
                "nps_score": self.nps_score,
                **(self.context or {})
            }
        }


class CustomerSignal(Base):
    """
    Customer signals/events for time-series analysis.

    Stores individual signals that are aggregated for intelligence.
    Designed for TimescaleDB hypertable conversion.
    """
    __tablename__ = "harvester_signals"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign keys
    customer_id = Column(String(100), ForeignKey("harvester_customers.customer_id"), nullable=False)
    tenant_id = Column(String(50), nullable=False, index=True)

    # Signal information
    signal_type = Column(SQLEnum(SignalType), nullable=False)
    signal_name = Column(String(100), nullable=False)
    severity = Column(SQLEnum(SignalSeverity), default=SignalSeverity.MEDIUM)

    # Signal value and context
    value = Column(Float)  # Numeric value if applicable
    text_value = Column(Text)  # Text value if applicable
    metadata = Column(JSONB, default={})

    # Source information
    source = Column(String(50))  # kafka, api, crm, oss, etc.
    source_event_id = Column(String(100))  # Original event ID
    channel = Column(String(50))  # phone, email, chat, web

    # Timestamps
    event_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    ingested_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    customer = relationship("CustomerProfile", back_populates="signals")

    # Indexes for time-series queries
    __table_args__ = (
        Index("ix_harvester_signals_tenant_time", "tenant_id", "event_time"),
        Index("ix_harvester_signals_customer_time", "customer_id", "event_time"),
        Index("ix_harvester_signals_type_severity", "signal_type", "severity"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "signal_name": self.signal_name,
            "severity": self.severity.value,
            "value": self.value,
            "text_value": self.text_value,
            "channel": self.channel,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "metadata": self.metadata
        }


class IntelligenceResult(Base):
    """
    Intelligence pipeline results.

    Stores the output of HarvesterV2 analysis for each customer.
    """
    __tablename__ = "harvester_results"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign keys
    customer_id = Column(String(100), ForeignKey("harvester_customers.customer_id"), nullable=False)
    tenant_id = Column(String(50), nullable=False, index=True)

    # Pipeline execution info
    workflow_id = Column(String(100), nullable=False, index=True)
    pipeline_version = Column(String(20), default="v2")

    # Scores from analysis
    churn_risk = Column(Float)
    fraud_risk = Column(Float)
    propensity_score = Column(Float)
    opportunity_score = Column(Float)
    aggregate_risk_score = Column(Float)
    critique_score = Column(Float)

    # Full results as JSONB
    scoring_result = Column(JSONB)
    causality_result = Column(JSONB)
    research_result = Column(JSONB)
    opportunity_result = Column(JSONB)
    risk_signal_result = Column(JSONB)
    nba_result = Column(JSONB)
    retention_result = Column(JSONB)
    final_intelligence = Column(JSONB)

    # Execution metadata
    execution_time_ms = Column(Integer)
    iterations = Column(Integer, default=1)
    subagents_executed = Column(JSONB)  # List of subagent names

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    customer = relationship("CustomerProfile", back_populates="results")
    actions = relationship("RecommendedAction", back_populates="result", lazy="dynamic")

    # Indexes
    __table_args__ = (
        Index("ix_harvester_results_tenant_time", "tenant_id", "created_at"),
        Index("ix_harvester_results_workflow", "workflow_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "customer_id": self.customer_id,
            "scores": {
                "churn_risk": self.churn_risk,
                "fraud_risk": self.fraud_risk,
                "propensity_score": self.propensity_score,
                "opportunity_score": self.opportunity_score,
                "aggregate_risk_score": self.aggregate_risk_score,
                "critique_score": self.critique_score
            },
            "scoring_result": self.scoring_result,
            "causality_result": self.causality_result,
            "research_result": self.research_result,
            "opportunity_result": self.opportunity_result,
            "risk_signal_result": self.risk_signal_result,
            "nba_result": self.nba_result,
            "retention_result": self.retention_result,
            "final_intelligence": self.final_intelligence,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class RecommendedAction(Base):
    """
    Recommended actions from intelligence analysis.

    Tracks NBA and retention actions with their execution status.
    """
    __tablename__ = "harvester_actions"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign keys
    customer_id = Column(String(100), ForeignKey("harvester_customers.customer_id"), nullable=False)
    tenant_id = Column(String(50), nullable=False, index=True)
    result_id = Column(Integer, ForeignKey("harvester_results.id"))

    # Action details
    action_type = Column(String(50), nullable=False)  # nba, retention, upsell, etc.
    action_name = Column(String(255), nullable=False)
    description = Column(Text)
    priority = Column(Integer, default=5)  # 1=highest, 10=lowest
    urgency = Column(String(20))  # immediate, today, this_week, this_month

    # Action content
    script = Column(Text)  # Agent script if applicable
    offer_details = Column(JSONB)  # Offer/discount details

    # Status tracking
    status = Column(SQLEnum(ActionStatus), default=ActionStatus.PENDING)
    assigned_to = Column(String(100))  # Agent ID

    # Outcome tracking
    outcome = Column(String(50))  # accepted, rejected, no_response, etc.
    outcome_notes = Column(Text)
    revenue_impact = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    due_date = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationships
    customer = relationship("CustomerProfile", back_populates="actions")
    result = relationship("IntelligenceResult", back_populates="actions")

    # Indexes
    __table_args__ = (
        Index("ix_harvester_actions_tenant_status", "tenant_id", "status"),
        Index("ix_harvester_actions_priority", "tenant_id", "priority", "status"),
        Index("ix_harvester_actions_assigned", "assigned_to", "status"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "action_type": self.action_type,
            "action_name": self.action_name,
            "description": self.description,
            "priority": self.priority,
            "urgency": self.urgency,
            "status": self.status.value if self.status else None,
            "assigned_to": self.assigned_to,
            "outcome": self.outcome,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None
        }
