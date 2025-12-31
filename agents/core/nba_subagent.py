"""
NBA (Next Best Action) Sub-Agent for Harvester V2
===================================================

Generates personalized next best actions with talk tracks:
- Action type selection (retention, upsell, cross-sell, service)
- Expected value calculation
- Personalized script generation
- Objection handlers

Usage:
    subagent = NBASubAgent()
    result = await subagent.run_isolated(
        context,
        "Generate next best action"
    )
"""

from typing import Any, Dict
import json

from harvester_v2.agents.base_subagent import BaseSubAgent, SubAgentContext
from harvester_v2.agents.parallel_executor import SubAgentRegistry
from shared.ai.model_router import TaskType
from shared.logging import get_platform_logger

logger = get_platform_logger(__name__)


@SubAgentRegistry.register
class NBASubAgent(BaseSubAgent):
    """
    NBA sub-agent for next best action generation.

    Outputs:
    1. action_type: Type of action to take
    2. action_name: Specific action name
    3. expected_value: Monetary impact estimate
    4. script: Personalized talk track
    5. objection_handlers: Responses to likely objections
    """

    agent_name = "nba_subagent"

    def __init__(self, options: Dict[str, Any] = None):
        super().__init__(agent_name="nba_subagent", options=options or {})

    async def _plan(self, context: SubAgentContext, task: str) -> Dict[str, Any]:
        """Plan NBA generation based on scores and context."""
        scores = context.input_data.get("scores", {})

        return {
            "should_recommend_retention": scores.get("churn_risk", 0) > 0.5,
            "should_recommend_upsell": scores.get("propensity_score", 0) > 0.6,
            "should_block_action": scores.get("fraud_score", 0) > 0.8,
            "customer_segment": scores.get("customer_segment", "unknown"),
            "available_context": list(context.input_data.keys())
        }

    async def _execute_plan(self, context: SubAgentContext, plan: Dict) -> Dict[str, Any]:
        """Generate NBA with personalized script."""
        prompt = f"""You are generating a Next Best Action (NBA) recommendation.

CUSTOMER DATA:
{json.dumps(context.input_data, indent=2, default=str)}

PLAN:
{json.dumps(plan, indent=2, default=str)}

RELEVANT MEMORY:
{json.dumps(context.memory_snapshot, indent=2, default=str) if context.memory_snapshot else "(none)"}

Generate the SINGLE BEST action for this customer.

Action types to consider:
- retention: Keep customer from churning (discounts, upgrades, attention)
- upsell: Sell higher-tier product
- cross_sell: Sell complementary product
- service: Proactive service improvement
- escalation: Route to specialist/manager

Script should be:
- Personalized to customer context
- Empathetic and value-focused
- Include specific offer details
- Handle likely objections

Respond with ONLY a JSON object (no markdown):
{{
    "action_type": "retention|upsell|cross_sell|service|escalation",
    "action_name": "Specific action name (e.g., 'Offer 20% Loyalty Discount')",
    "expected_value": 150.00,
    "confidence": 0.X,
    "urgency": "immediate|24h|7d|30d",
    "reasoning": "Why this is the best action for this customer",

    "script": {{
        "opening": "Personalized opening line acknowledging customer",
        "value_proposition": "Clear statement of what we're offering and why it benefits them",
        "key_points": ["Point 1", "Point 2", "Point 3"],
        "objection_handlers": [
            {{"objection": "Likely objection 1", "response": "How to address it"}},
            {{"objection": "Likely objection 2", "response": "How to address it"}}
        ],
        "closing": "Clear call to action"
    }},

    "offer_details": {{
        "product": "Product or service name",
        "discount_percent": 0,
        "discount_amount": 0.00,
        "contract_months": 0,
        "expires_in_days": 7
    }},

    "alternative_actions": [
        {{"action": "Backup action if primary fails", "confidence": 0.X}}
    ],

    "blockers": ["Any reasons NOT to take this action"],

    "success_metrics": {{
        "primary": "How we measure success",
        "target": "Target value"
    }}
}}"""

        try:
            response = await self.model_router.complete(
                prompt=prompt,
                task_type=TaskType.REASONING,
                agent_name=self.agent_name,
                tenant_id=context.tenant_id
            )

            response_text = response.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            result = json.loads(response_text)
            result["customer_id"] = context.customer_id
            result["tenant_id"] = context.tenant_id
            return result

        except json.JSONDecodeError as e:
            logger.warning(
                "NBA response JSON parsing failed, using fallback",
                error=str(e),
                customer_id=context.customer_id
            )
            return self._get_fallback_nba(context, f"JSON parse error: {e}")

        except Exception as e:
            logger.error("NBA generation failed", error=str(e))
            return self._get_fallback_nba(context, str(e))

    def _get_fallback_nba(self, context: SubAgentContext, error: str) -> Dict[str, Any]:
        """Return fallback NBA when generation fails."""
        return {
            "action_type": "service",
            "action_name": "General service check-in",
            "expected_value": 0,
            "confidence": 0,
            "urgency": "7d",
            "reasoning": "NBA generation failed, defaulting to service check-in",
            "script": {"opening": "Hello", "closing": "Thank you"},
            "error": error,
            "customer_id": context.customer_id
        }
