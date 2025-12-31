"""
Opportunity Sub-Agent for Harvester V2
=======================================

Identifies opportunities for upsell, cross-sell, and account expansion
by synthesizing signals from multiple sources:

- Scoring signals (propensity scores, usage patterns)
- Research findings (market trends, competitor gaps)
- Causality insights (what would delight this customer)
- Customer context (industry, segment, tenure)

This sub-agent finds the "opportunity signals" that complement
risk signals for a complete customer intelligence picture.

Usage:
    subagent = OpportunitySubAgent()
    result = await subagent.run_isolated(
        context,
        "Identify opportunities for customer growth"
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
class OpportunitySubAgent(BaseSubAgent):
    """
    Opportunity detection sub-agent.

    Identifies growth opportunities by cross-referencing:
    1. Customer propensity scores
    2. Usage patterns and trends
    3. Market research findings
    4. Competitive landscape gaps
    5. Customer lifecycle stage

    Outputs:
    - opportunities: List of identified opportunities
    - opportunity_score: Overall opportunity potential (0-1)
    - prioritized_actions: Ranked list of growth actions
    - timing_recommendation: When to act on opportunities
    """

    agent_name = "opportunity_subagent"

    # Standard critique threshold
    critique_threshold = 0.85

    def __init__(self, options: Dict[str, Any] = None):
        super().__init__(agent_name="opportunity_subagent", options=options or {})

    async def _plan(self, context: SubAgentContext, task: str) -> Dict[str, Any]:
        """Plan opportunity analysis by identifying relevant signals."""
        input_data = context.input_data or {}

        # Extract available signals
        available_signals = {
            "has_propensity": "propensity_score" in input_data,
            "has_usage": "usage_trend" in input_data,
            "has_churn_risk": "churn_risk" in input_data,
            "has_research": "research_synthesis" in input_data or "research_result" in input_data,
            "has_segment": "customer_segment" in input_data,
            "has_revenue": "monthly_revenue" in input_data,
            "has_product": "product" in input_data,
            "has_industry": "industry" in input_data,
            "has_tenure": "tenure_months" in input_data
        }

        # Determine analysis approach based on available data
        analysis_focus = []

        if available_signals["has_propensity"]:
            analysis_focus.append("propensity_based_upsell")

        if available_signals["has_usage"]:
            analysis_focus.append("usage_pattern_opportunities")

        if available_signals["has_research"]:
            analysis_focus.append("market_driven_opportunities")

        if available_signals["has_segment"] and available_signals["has_revenue"]:
            analysis_focus.append("segment_expansion")

        if available_signals["has_industry"]:
            analysis_focus.append("industry_specific_offerings")

        # If churn risk is low, focus on growth; if high, focus on retention upsell
        if available_signals["has_churn_risk"]:
            churn_risk = input_data.get("churn_risk", 0.5)
            if churn_risk < 0.3:
                analysis_focus.append("aggressive_growth")
            elif churn_risk < 0.7:
                analysis_focus.append("balanced_growth_retention")
            else:
                analysis_focus.append("retention_first_upsell")

        return {
            "available_signals": available_signals,
            "analysis_focus": analysis_focus,
            "customer_context": {
                "customer_id": context.customer_id,
                "segment": input_data.get("customer_segment", "unknown"),
                "tenure_months": input_data.get("tenure_months", 0),
                "monthly_revenue": input_data.get("monthly_revenue", 0)
            }
        }

    async def _execute_plan(self, context: SubAgentContext, plan: Dict) -> Dict[str, Any]:
        """Execute opportunity analysis using LLM reasoning."""
        input_data = context.input_data or {}
        analysis_focus = plan.get("analysis_focus", [])

        # Build comprehensive prompt for opportunity identification
        prompt = f"""You are an opportunity analyst for customer intelligence.
Identify growth opportunities for this customer based on available data.

CUSTOMER CONTEXT:
- Customer ID: {context.customer_id}
- Industry: {input_data.get('industry', 'Unknown')}
- Product: {input_data.get('product', 'Unknown')}
- Monthly Revenue: ${input_data.get('monthly_revenue', 0):,}
- Tenure: {input_data.get('tenure_months', 0)} months
- Segment: {input_data.get('customer_segment', 'Unknown')}

SCORING SIGNALS:
- Propensity Score: {input_data.get('propensity_score', 'N/A')}
- Churn Risk: {input_data.get('churn_risk', 'N/A')}
- Usage Trend: {input_data.get('usage_trend', 'N/A')}

RESEARCH FINDINGS:
{json.dumps(input_data.get('research_synthesis', input_data.get('research_result', {})), indent=2, default=str)[:2000]}

CAUSALITY INSIGHTS:
{json.dumps(input_data.get('causality_result', {}), indent=2, default=str)[:1500]}

ANALYSIS FOCUS AREAS: {analysis_focus}

Identify opportunities across these categories:
1. UPSELL: Higher-tier products or expanded capacity
2. CROSS-SELL: Complementary products or services
3. EXPANSION: Geographic, departmental, or usage expansion
4. VALUE-ADD: Premium features, support tiers, or partnerships
5. STRATEGIC: Long-term relationship deepening

Respond with ONLY a JSON object (no markdown):
{{
    "opportunities": [
        {{
            "type": "upsell|cross_sell|expansion|value_add|strategic",
            "name": "Opportunity name",
            "description": "What the opportunity is",
            "potential_revenue_impact": "low|medium|high|very_high",
            "confidence": 0.X,
            "signals_supporting": ["signal1", "signal2"],
            "timing": "immediate|short_term|medium_term|long_term",
            "prerequisites": ["any conditions that must be met first"],
            "recommended_approach": "How to pursue this opportunity"
        }}
    ],
    "opportunity_score": 0.X,
    "prioritized_actions": [
        {{
            "rank": 1,
            "action": "Specific action to take",
            "opportunity_type": "upsell|cross_sell|etc",
            "expected_outcome": "What success looks like",
            "urgency": "immediate|this_week|this_month|this_quarter"
        }}
    ],
    "timing_recommendation": {{
        "optimal_window": "When to act",
        "blocking_factors": ["Any factors preventing immediate action"],
        "enabling_factors": ["Factors that support action now"]
    }},
    "analysis_summary": "2-3 sentence summary of overall opportunity landscape"
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
            result["analysis_focus"] = analysis_focus

            return result

        except json.JSONDecodeError as e:
            logger.warning(
                "Opportunity analysis JSON parsing failed",
                error=str(e),
                customer_id=context.customer_id
            )
            return self._get_fallback_result(context, f"JSON parsing failed: {e}")

        except Exception as e:
            logger.error("Opportunity analysis failed", error=str(e))
            return self._get_fallback_result(context, str(e))

    def _get_fallback_result(
        self,
        context: SubAgentContext,
        error: str
    ) -> Dict[str, Any]:
        """Return fallback result when analysis fails."""
        input_data = context.input_data or {}

        # Generate basic opportunities based on raw signals
        opportunities = []

        # If propensity is decent, suggest upsell
        if input_data.get("propensity_score", 0) > 0.5:
            opportunities.append({
                "type": "upsell",
                "name": "Capacity Expansion",
                "description": "Customer shows positive propensity signals",
                "potential_revenue_impact": "medium",
                "confidence": 0.5,
                "signals_supporting": ["propensity_score"],
                "timing": "short_term",
                "prerequisites": [],
                "recommended_approach": "Present expansion options"
            })

        # If usage is growing, suggest tier upgrade
        if input_data.get("usage_trend") == "growing":
            opportunities.append({
                "type": "upsell",
                "name": "Tier Upgrade",
                "description": "Usage trend indicates growth",
                "potential_revenue_impact": "high",
                "confidence": 0.6,
                "signals_supporting": ["usage_trend"],
                "timing": "immediate",
                "prerequisites": [],
                "recommended_approach": "Propose higher tier"
            })

        return {
            "opportunities": opportunities,
            "opportunity_score": 0.3 if opportunities else 0.0,
            "prioritized_actions": [],
            "timing_recommendation": {
                "optimal_window": "Unknown",
                "blocking_factors": [error],
                "enabling_factors": []
            },
            "analysis_summary": f"Fallback analysis due to: {error}",
            "customer_id": context.customer_id,
            "tenant_id": context.tenant_id,
            "error": error
        }
