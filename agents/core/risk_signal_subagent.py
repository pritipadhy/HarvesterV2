"""
Risk Signal Sub-Agent for Harvester V2
=======================================

Synthesizes all risk signals into a unified risk profile by aggregating:

- Churn risk (from ScoringSubAgent)
- Fraud risk (from ScoringSubAgent)
- Compliance risk (from context and history)
- Competitive risk (from ResearchSubAgent)
- Operational risk (from support patterns)
- Financial risk (from payment history)

This sub-agent finds the "risk signals" that complement opportunity
signals for a complete customer intelligence picture.

Usage:
    subagent = RiskSignalSubAgent()
    result = await subagent.run_isolated(
        context,
        "Aggregate risk signals for customer"
    )
"""

from typing import Any, Dict, List, Optional
import json

from harvester_v2.agents.base_subagent import BaseSubAgent, SubAgentContext
from harvester_v2.agents.parallel_executor import SubAgentRegistry
from shared.ai.model_router import TaskType
from shared.logging import get_platform_logger

logger = get_platform_logger(__name__)


@SubAgentRegistry.register
class RiskSignalSubAgent(BaseSubAgent):
    """
    Risk signal aggregation sub-agent.

    Synthesizes risk signals from multiple sources into a unified
    risk profile with prioritized alerts and recommended mitigations.

    Risk Categories:
    1. CHURN: Customer leaving risk
    2. FRAUD: Fraudulent activity risk
    3. COMPETITIVE: Competitor threat risk
    4. COMPLIANCE: Regulatory/policy risk
    5. OPERATIONAL: Service/support risk
    6. FINANCIAL: Payment/revenue risk

    Outputs:
    - risk_signals: List of identified risk signals
    - aggregate_risk_score: Overall risk level (0-1)
    - risk_profile: Detailed breakdown by category
    - priority_alerts: Urgent items requiring immediate attention
    - recommended_mitigations: Actions to reduce risk
    """

    agent_name = "risk_signal_subagent"

    # Standard critique threshold
    critique_threshold = 0.85

    # Risk severity thresholds
    SEVERITY_THRESHOLDS = {
        "critical": 0.85,
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3
    }

    def __init__(self, options: Dict[str, Any] = None):
        super().__init__(agent_name="risk_signal_subagent", options=options or {})

    async def _plan(self, context: SubAgentContext, task: str) -> Dict[str, Any]:
        """Plan risk aggregation by identifying available risk signals."""
        input_data = context.input_data or {}

        # Identify available risk signals
        available_signals = {
            # Scoring-based risks
            "churn_risk": input_data.get("churn_risk"),
            "fraud_score": input_data.get("fraud_score"),
            "risk_score": input_data.get("risk_score"),

            # Context-based risks
            "competitor_mentions": input_data.get("competitor_mentions", []),
            "support_tickets": input_data.get("support_tickets_last_90d"),
            "nps_score": input_data.get("nps_score"),
            "usage_trend": input_data.get("usage_trend"),
            "payment_history": input_data.get("payment_history"),
            "contract_end_date": input_data.get("contract_end_date"),

            # Analysis-based risks
            "causality_result": input_data.get("causality_result", {}),
            "research_result": input_data.get("research_result", {}),
            "recent_interactions": input_data.get("recent_interactions", [])
        }

        # Determine which risk categories to analyze
        risk_categories = []

        if available_signals["churn_risk"] is not None:
            risk_categories.append("churn")

        if available_signals["fraud_score"] is not None:
            risk_categories.append("fraud")

        if available_signals["competitor_mentions"]:
            risk_categories.append("competitive")

        if available_signals["support_tickets"] or available_signals["recent_interactions"]:
            risk_categories.append("operational")

        if available_signals["payment_history"]:
            risk_categories.append("financial")

        # Always include compliance as a general check
        risk_categories.append("compliance")

        return {
            "available_signals": {k: v is not None and v != [] for k, v in available_signals.items()},
            "risk_categories": risk_categories,
            "raw_signals": available_signals,
            "customer_context": {
                "customer_id": context.customer_id,
                "segment": input_data.get("customer_segment", "unknown"),
                "tenure_months": input_data.get("tenure_months", 0)
            }
        }

    async def _execute_plan(self, context: SubAgentContext, plan: Dict) -> Dict[str, Any]:
        """Execute risk aggregation using LLM reasoning."""
        input_data = context.input_data or {}
        raw_signals = plan.get("raw_signals", {})
        risk_categories = plan.get("risk_categories", [])

        # Pre-compute some risk signals for efficiency
        pre_computed_signals = self._compute_basic_signals(raw_signals)

        # Build comprehensive prompt for risk synthesis
        prompt = f"""You are a risk analyst synthesizing customer risk signals into a unified profile.

CUSTOMER CONTEXT:
- Customer ID: {context.customer_id}
- Industry: {input_data.get('industry', 'Unknown')}
- Segment: {input_data.get('customer_segment', 'Unknown')}
- Tenure: {input_data.get('tenure_months', 0)} months
- Monthly Revenue: ${input_data.get('monthly_revenue', 0):,}
- Contract End: {input_data.get('contract_end_date', 'Unknown')}

SCORING SIGNALS:
- Churn Risk: {raw_signals.get('churn_risk', 'N/A')}
- Fraud Score: {raw_signals.get('fraud_score', 'N/A')}
- Overall Risk Score: {raw_signals.get('risk_score', 'N/A')}

BEHAVIORAL SIGNALS:
- NPS Score: {raw_signals.get('nps_score', 'N/A')}
- Support Tickets (90d): {raw_signals.get('support_tickets', 'N/A')}
- Usage Trend: {raw_signals.get('usage_trend', 'N/A')}
- Payment History: {raw_signals.get('payment_history', 'N/A')}

COMPETITIVE INTELLIGENCE:
- Competitor Mentions: {raw_signals.get('competitor_mentions', [])}

RECENT INTERACTIONS:
{json.dumps(raw_signals.get('recent_interactions', [])[:5], indent=2, default=str)}

CAUSALITY ANALYSIS:
{json.dumps(raw_signals.get('causality_result', {}), indent=2, default=str)[:1500]}

RESEARCH FINDINGS:
{json.dumps(raw_signals.get('research_result', {}), indent=2, default=str)[:1000]}

PRE-COMPUTED SIGNALS:
{json.dumps(pre_computed_signals, indent=2, default=str)}

RISK CATEGORIES TO ANALYZE: {risk_categories}

Synthesize all signals into a unified risk profile.
For each risk signal:
- Identify the source(s) of the signal
- Assess severity (critical/high/medium/low)
- Determine urgency (immediate/this_week/this_month/monitor)
- Suggest mitigations

Respond with ONLY a JSON object (no markdown):
{{
    "risk_signals": [
        {{
            "type": "churn|fraud|competitive|compliance|operational|financial",
            "name": "Signal name",
            "description": "What this risk signal indicates",
            "severity": "critical|high|medium|low",
            "confidence": 0.X,
            "sources": ["where this signal comes from"],
            "urgency": "immediate|this_week|this_month|monitor",
            "impact": "What happens if not addressed",
            "recommended_mitigation": "How to reduce this risk"
        }}
    ],
    "aggregate_risk_score": 0.X,
    "risk_profile": {{
        "churn": {{"score": 0.X, "severity": "...", "trend": "increasing|stable|decreasing"}},
        "fraud": {{"score": 0.X, "severity": "...", "trend": "..."}},
        "competitive": {{"score": 0.X, "severity": "...", "trend": "..."}},
        "compliance": {{"score": 0.X, "severity": "...", "trend": "..."}},
        "operational": {{"score": 0.X, "severity": "...", "trend": "..."}},
        "financial": {{"score": 0.X, "severity": "...", "trend": "..."}}
    }},
    "priority_alerts": [
        {{
            "alert_type": "churn|fraud|etc",
            "severity": "critical|high",
            "message": "Urgent alert message",
            "action_required": "What must be done immediately"
        }}
    ],
    "recommended_mitigations": [
        {{
            "rank": 1,
            "risk_type": "churn|fraud|etc",
            "mitigation": "Specific action to take",
            "expected_risk_reduction": 0.X,
            "effort": "low|medium|high",
            "timeline": "How long this takes"
        }}
    ],
    "risk_summary": "2-3 sentence summary of overall risk posture"
}}"""

        try:
            response = await self.model_router.complete(
                prompt=prompt,
                task_type=TaskType.REASONING,
                agent_name=f"{self.agent_name}_analyzer",
                tenant_id=context.tenant_id
            )

            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            result = json.loads(response_text)

            # Add metadata
            result["customer_id"] = context.customer_id
            result["tenant_id"] = context.tenant_id
            result["risk_categories_analyzed"] = risk_categories

            return result

        except json.JSONDecodeError as e:
            logger.warning(
                "Risk signal JSON parsing failed",
                error=str(e),
                customer_id=context.customer_id
            )
            return self._get_fallback_result(context, plan, f"JSON parsing failed: {e}")

        except Exception as e:
            logger.error("Risk signal aggregation failed", error=str(e))
            return self._get_fallback_result(context, plan, str(e))

    def _compute_basic_signals(self, raw_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Compute basic risk signals from raw data."""
        computed = {}

        # Churn risk severity
        churn_risk = raw_signals.get("churn_risk")
        if churn_risk is not None:
            if churn_risk >= self.SEVERITY_THRESHOLDS["critical"]:
                computed["churn_severity"] = "critical"
            elif churn_risk >= self.SEVERITY_THRESHOLDS["high"]:
                computed["churn_severity"] = "high"
            elif churn_risk >= self.SEVERITY_THRESHOLDS["medium"]:
                computed["churn_severity"] = "medium"
            else:
                computed["churn_severity"] = "low"

        # NPS-based risk
        nps = raw_signals.get("nps_score")
        if nps is not None:
            if nps <= 6:
                computed["nps_risk"] = "high"  # Detractor
            elif nps <= 8:
                computed["nps_risk"] = "medium"  # Passive
            else:
                computed["nps_risk"] = "low"  # Promoter

        # Support ticket risk
        tickets = raw_signals.get("support_tickets")
        if tickets is not None:
            if tickets >= 10:
                computed["support_risk"] = "high"
            elif tickets >= 5:
                computed["support_risk"] = "medium"
            else:
                computed["support_risk"] = "low"

        # Usage trend risk
        usage = raw_signals.get("usage_trend")
        if usage:
            if usage == "declining":
                computed["usage_risk"] = "high"
            elif usage == "stable":
                computed["usage_risk"] = "low"
            else:
                computed["usage_risk"] = "very_low"

        # Competitor risk
        competitors = raw_signals.get("competitor_mentions", [])
        if competitors:
            computed["competitor_risk"] = "high" if len(competitors) > 1 else "medium"
        else:
            computed["competitor_risk"] = "low"

        # Negative interaction ratio
        interactions = raw_signals.get("recent_interactions", [])
        if interactions:
            negative = sum(1 for i in interactions if i.get("sentiment") == "negative")
            ratio = negative / len(interactions)
            computed["negative_interaction_ratio"] = round(ratio, 2)
            computed["interaction_risk"] = "high" if ratio > 0.5 else ("medium" if ratio > 0.25 else "low")

        return computed

    def _get_fallback_result(
        self,
        context: SubAgentContext,
        plan: Dict[str, Any],
        error: str
    ) -> Dict[str, Any]:
        """Return fallback result based on pre-computed signals."""
        raw_signals = plan.get("raw_signals", {})
        pre_computed = self._compute_basic_signals(raw_signals)

        # Generate basic risk signals from pre-computed data
        risk_signals = []

        # Churn risk signal
        churn_risk = raw_signals.get("churn_risk", 0)
        if churn_risk > 0.5:
            risk_signals.append({
                "type": "churn",
                "name": "Elevated Churn Risk",
                "description": f"Churn risk score of {churn_risk:.2f} indicates customer may leave",
                "severity": pre_computed.get("churn_severity", "medium"),
                "confidence": 0.8,
                "sources": ["scoring_result"],
                "urgency": "immediate" if churn_risk > 0.8 else "this_week",
                "impact": "Revenue loss if customer churns",
                "recommended_mitigation": "Initiate retention protocol"
            })

        # Competitive risk signal
        if raw_signals.get("competitor_mentions"):
            risk_signals.append({
                "type": "competitive",
                "name": "Competitor Evaluation",
                "description": f"Customer mentioned competitors: {raw_signals['competitor_mentions']}",
                "severity": "high",
                "confidence": 0.9,
                "sources": ["customer_context"],
                "urgency": "this_week",
                "impact": "Customer may switch to competitor",
                "recommended_mitigation": "Prepare competitive counter-offer"
            })

        # Calculate aggregate risk
        aggregate = 0.0
        if churn_risk:
            aggregate = max(aggregate, churn_risk)
        if raw_signals.get("fraud_score"):
            aggregate = max(aggregate, raw_signals["fraud_score"])

        return {
            "risk_signals": risk_signals,
            "aggregate_risk_score": aggregate,
            "risk_profile": {
                "churn": {"score": churn_risk or 0, "severity": pre_computed.get("churn_severity", "low"), "trend": "unknown"},
                "fraud": {"score": raw_signals.get("fraud_score", 0), "severity": "low", "trend": "unknown"},
                "competitive": {"score": 0.7 if raw_signals.get("competitor_mentions") else 0.1, "severity": pre_computed.get("competitor_risk", "low"), "trend": "unknown"},
                "compliance": {"score": 0.1, "severity": "low", "trend": "stable"},
                "operational": {"score": 0.3 if pre_computed.get("support_risk") == "high" else 0.1, "severity": pre_computed.get("support_risk", "low"), "trend": "unknown"},
                "financial": {"score": 0.1, "severity": "low", "trend": "stable"}
            },
            "priority_alerts": [
                {
                    "alert_type": "churn",
                    "severity": "high" if churn_risk > 0.7 else "medium",
                    "message": "Customer shows elevated churn risk",
                    "action_required": "Review account immediately"
                }
            ] if churn_risk > 0.5 else [],
            "recommended_mitigations": [],
            "risk_summary": f"Fallback analysis due to: {error}",
            "customer_id": context.customer_id,
            "tenant_id": context.tenant_id,
            "error": error
        }
