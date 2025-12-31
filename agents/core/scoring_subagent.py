"""
Scoring Sub-Agent for Harvester V2
===================================

Calculates customer risk scores using LLM reasoning:
- Fraud score (0-1): Risk of fraudulent activity
- Churn risk (0-1): Probability of customer churning
- Propensity score (0-1): Likelihood of upsell/cross-sell
- Risk score (0-1): Composite risk assessment

The scoring sub-agent uses the DeepAgents pattern:
1. PLAN: Identify relevant data signals for each score
2. EXECUTE: Calculate scores with reasoning
3. CRITIQUE: Evaluate score quality
4. OUTPUT: Return structured scoring result

Usage:
    subagent = ScoringSubAgent()
    result = await subagent.run_isolated(
        context,
        "Calculate customer risk scores"
    )

    # Access scores
    fraud_score = result.data.get("fraud_score")
    churn_risk = result.data.get("churn_risk")
"""

from typing import Any, Dict
import json
import structlog

from harvester_v2.agents.base_subagent import (
    BaseSubAgent,
    SubAgentContext,
    SubAgentResult
)
from harvester_v2.agents.parallel_executor import SubAgentRegistry
from shared.ai.model_router import get_model_router, TaskType
from shared.logging import get_platform_logger

logger = get_platform_logger(__name__)


@SubAgentRegistry.register
class ScoringSubAgent(BaseSubAgent):
    """
    Scoring sub-agent for customer risk assessment.

    Calculates four key scores:
    1. fraud_score: Risk of fraudulent activity (0-1, higher = more risk)
    2. churn_risk: Probability of churning (0-1, higher = more likely)
    3. propensity_score: Upsell/cross-sell opportunity (0-1, higher = better)
    4. risk_score: Composite risk (0-1, higher = more risk)

    Each score includes:
    - Numeric value
    - Contributing factors
    - Reasoning explanation
    - Confidence level
    """

    # Class-level agent name for registry
    agent_name = "scoring_subagent"

    def __init__(self, options: Dict[str, Any] = None):
        """
        Initialize scoring sub-agent.

        Args:
            options: Optional configuration overrides
        """
        super().__init__(
            agent_name="scoring_subagent",
            options=options or {}
        )

        # Scoring thresholds (configurable per tenant)
        self.high_risk_threshold = self.options.get("high_risk_threshold", 0.7)
        self.high_churn_threshold = self.options.get("high_churn_threshold", 0.6)
        self.high_propensity_threshold = self.options.get("high_propensity_threshold", 0.6)

    async def _plan(
        self,
        context: SubAgentContext,
        task: str
    ) -> Dict[str, Any]:
        """
        Plan scoring approach based on available data.

        Analyzes input data to determine:
        - Which scores to calculate
        - What signals are available
        - Expected output format
        """
        prompt = f"""You are planning a customer risk scoring analysis.

CUSTOMER DATA:
{json.dumps(context.input_data, indent=2, default=str)}

RELEVANT MEMORY:
{json.dumps(context.memory_snapshot, indent=2, default=str) if context.memory_snapshot else "(none)"}

Analyze the available data and plan which scores to calculate:

1. fraud_score - What signals indicate fraud risk?
   - Unusual activity patterns
   - Payment anomalies
   - Account age and behavior

2. churn_risk - What signals indicate churn likelihood?
   - Usage patterns
   - Engagement level
   - Service issues
   - Tenure and contract status

3. propensity_score - What signals indicate upsell opportunity?
   - Current product usage
   - Engagement with features
   - Customer segment

4. risk_score - What composite risk factors exist?
   - Overall health
   - Payment history
   - Support interactions

Respond with ONLY a JSON object (no markdown):
{{
    "scores_to_calculate": ["fraud", "churn", "propensity", "risk"],
    "fraud_signals": ["list of relevant data points for fraud"],
    "churn_signals": ["list of relevant data points for churn"],
    "propensity_signals": ["list of relevant data points for propensity"],
    "risk_signals": ["list of relevant data points for risk"],
    "data_quality_notes": "Any concerns about data quality or missing signals"
}}"""

        try:
            response = await self.model_router.complete(
                prompt=prompt,
                task_type=TaskType.REASONING,
                agent_name=self.agent_name,
                tenant_id=context.tenant_id
            )

            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            plan = json.loads(response_text)
            return plan

        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse plan JSON, using default",
                response=response[:200] if response else ""
            )
            return {
                "scores_to_calculate": ["fraud", "churn", "propensity", "risk"],
                "fraud_signals": [],
                "churn_signals": [],
                "propensity_signals": [],
                "risk_signals": [],
                "data_quality_notes": "Plan parsing failed, using defaults"
            }

        except Exception as e:
            logger.warning("Planning failed", error=str(e))
            return {"scores_to_calculate": ["fraud", "churn", "propensity", "risk"]}

    async def _execute_plan(
        self,
        context: SubAgentContext,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate all scores based on plan.

        Uses LLM reasoning to analyze data and produce scores
        with explanations and contributing factors.
        """
        prompt = f"""You are calculating customer risk scores based on available data.

CUSTOMER DATA:
{json.dumps(context.input_data, indent=2, default=str)}

SCORING PLAN:
{json.dumps(plan, indent=2, default=str)}

RELEVANT MEMORY:
{json.dumps(context.memory_snapshot, indent=2, default=str) if context.memory_snapshot else "(none)"}

Calculate each score with detailed reasoning.

IMPORTANT SCORING GUIDELINES:
- All scores are 0.0 to 1.0
- Higher fraud_score = MORE fraud risk (bad)
- Higher churn_risk = MORE likely to churn (bad)
- Higher propensity_score = MORE likely to upsell (good)
- Higher risk_score = MORE overall risk (bad)

Respond with ONLY a JSON object (no markdown):
{{
    "fraud_score": 0.XX,
    "fraud_confidence": 0.XX,
    "fraud_reasoning": "Brief explanation of fraud risk assessment",
    "fraud_factors": ["factor1", "factor2"],

    "churn_risk": 0.XX,
    "churn_confidence": 0.XX,
    "churn_reasoning": "Brief explanation of churn risk assessment",
    "churn_factors": ["factor1", "factor2"],
    "churn_timeframe_days": 30,

    "propensity_score": 0.XX,
    "propensity_confidence": 0.XX,
    "propensity_reasoning": "Brief explanation of upsell opportunity",
    "propensity_opportunities": ["opportunity1", "opportunity2"],

    "risk_score": 0.XX,
    "risk_confidence": 0.XX,
    "risk_reasoning": "Brief explanation of overall risk",
    "risk_factors": ["factor1", "factor2"],

    "customer_segment": "high_value|at_risk|growth|nurture|watch",
    "segment_reasoning": "Why this segment was assigned",

    "scoring_summary": "One sentence overall assessment"
}}"""

        try:
            response = await self.model_router.complete(
                prompt=prompt,
                task_type=TaskType.REASONING,
                agent_name=self.agent_name,
                tenant_id=context.tenant_id
            )

            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            scores = json.loads(response_text)

            # Validate and normalize scores
            scores = self._normalize_scores(scores)

            # Add metadata
            scores["customer_id"] = context.customer_id
            scores["tenant_id"] = context.tenant_id
            scores["scoring_plan"] = plan

            return scores

        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse scores JSON",
                response=response[:200] if response else ""
            )
            return self._get_default_scores(context)

        except Exception as e:
            logger.error("Scoring execution failed", error=str(e))
            return self._get_default_scores(context, error=str(e))

    def _normalize_scores(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate all scores.

        Ensures all scores are within 0-1 range.
        """
        score_fields = [
            "fraud_score", "fraud_confidence",
            "churn_risk", "churn_confidence",
            "propensity_score", "propensity_confidence",
            "risk_score", "risk_confidence"
        ]

        for field in score_fields:
            if field in scores:
                try:
                    value = float(scores[field])
                    scores[field] = max(0.0, min(1.0, value))
                except (TypeError, ValueError):
                    scores[field] = 0.5  # Default to middle

        return scores

    def _get_default_scores(
        self,
        context: SubAgentContext,
        error: str = None
    ) -> Dict[str, Any]:
        """
        Get default scores when calculation fails.

        Returns neutral scores with error indication.
        """
        return {
            "fraud_score": 0.0,
            "fraud_confidence": 0.0,
            "fraud_reasoning": "Unable to calculate - insufficient data",
            "fraud_factors": [],

            "churn_risk": 0.5,
            "churn_confidence": 0.0,
            "churn_reasoning": "Unable to calculate - insufficient data",
            "churn_factors": [],
            "churn_timeframe_days": 0,

            "propensity_score": 0.5,
            "propensity_confidence": 0.0,
            "propensity_reasoning": "Unable to calculate - insufficient data",
            "propensity_opportunities": [],

            "risk_score": 0.5,
            "risk_confidence": 0.0,
            "risk_reasoning": "Unable to calculate - insufficient data",
            "risk_factors": [],

            "customer_segment": "unknown",
            "segment_reasoning": "Unable to determine segment",

            "scoring_summary": "Scoring failed - using default neutral scores",
            "error": error,
            "customer_id": context.customer_id,
            "tenant_id": context.tenant_id
        }

    def get_alerts(self, scores: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        Generate alerts based on scores.

        Returns list of alerts for high-risk situations.
        """
        alerts = []

        if scores.get("fraud_score", 0) >= self.high_risk_threshold:
            alerts.append({
                "type": "fraud_alert",
                "severity": "high",
                "score": scores.get("fraud_score"),
                "message": f"High fraud risk detected: {scores.get('fraud_reasoning', 'N/A')}",
                "factors": scores.get("fraud_factors", [])
            })

        if scores.get("churn_risk", 0) >= self.high_churn_threshold:
            alerts.append({
                "type": "churn_alert",
                "severity": "high" if scores["churn_risk"] >= 0.8 else "medium",
                "score": scores.get("churn_risk"),
                "message": f"High churn risk detected: {scores.get('churn_reasoning', 'N/A')}",
                "factors": scores.get("churn_factors", []),
                "timeframe_days": scores.get("churn_timeframe_days", 30)
            })

        if scores.get("propensity_score", 0) >= self.high_propensity_threshold:
            alerts.append({
                "type": "upsell_opportunity",
                "severity": "info",
                "score": scores.get("propensity_score"),
                "message": f"Upsell opportunity: {scores.get('propensity_reasoning', 'N/A')}",
                "opportunities": scores.get("propensity_opportunities", [])
            })

        return alerts
