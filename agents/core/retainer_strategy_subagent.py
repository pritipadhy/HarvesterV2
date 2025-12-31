"""
Retainer Strategy Sub-Agent for Harvester V2
==============================================

Generates comprehensive retention strategies for at-risk customers:
- Multi-touch retention campaigns
- Escalation paths
- Win-back strategies
- Negotiation tactics

Works alongside NBA sub-agent but focuses specifically on retention
with deeper strategy and longer-term planning.

Usage:
    subagent = RetainerStrategySubAgent()
    result = await subagent.run_isolated(
        context,
        "Generate retention strategy"
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
class RetainerStrategySubAgent(BaseSubAgent):
    """
    Retention strategy sub-agent.

    Focuses on comprehensive retention planning beyond single actions.

    Outputs:
    1. strategy_type: Overall retention approach
    2. campaign_phases: Multi-touch campaign plan
    3. escalation_path: When and how to escalate
    4. offers_sequence: Sequence of offers to try
    5. negotiation_limits: What we can offer
    6. win_back_plan: If they churn, how to win back
    """

    agent_name = "retainer_strategy_subagent"

    def __init__(self, options: Dict[str, Any] = None):
        super().__init__(agent_name="retainer_strategy_subagent", options=options or {})

    async def _plan(self, context: SubAgentContext, task: str) -> Dict[str, Any]:
        """Plan retention strategy approach."""
        scores = context.input_data.get("scores", {})
        churn_risk = scores.get("churn_risk", 0.5)

        return {
            "urgency_level": "critical" if churn_risk > 0.8 else ("high" if churn_risk > 0.6 else "moderate"),
            "churn_risk": churn_risk,
            "customer_value": scores.get("ltv", 0) or context.input_data.get("ltv", 0),
            "has_previous_retention": bool(context.memory_snapshot.get("previous_retention_attempts")),
            "available_offers": context.input_data.get("available_offers", [])
        }

    async def _execute_plan(self, context: SubAgentContext, plan: Dict) -> Dict[str, Any]:
        """Execute retention strategy generation."""
        prompt = f"""You are creating a comprehensive retention strategy for an at-risk customer.

CUSTOMER DATA:
{json.dumps(context.input_data, indent=2, default=str)}

STRATEGY PLAN:
{json.dumps(plan, indent=2, default=str)}

PREVIOUS INTERACTIONS:
{json.dumps(context.memory_snapshot, indent=2, default=str) if context.memory_snapshot else "(none)"}

Create a multi-phase retention strategy:

1. IMMEDIATE: What to do right now
2. FOLLOW-UP: If immediate doesn't work
3. ESCALATION: When to involve managers/specialists
4. LAST RESORT: Final offers before churn
5. WIN-BACK: If they do churn, how to get them back

Consider:
- Customer value (LTV)
- Churn urgency
- Previous attempts
- Budget constraints
- Brand reputation

Respond with ONLY a JSON object (no markdown):
{{
    "strategy_type": "proactive|reactive|urgent|winback",
    "urgency": "critical|high|moderate|low",
    "estimated_retention_probability": 0.X,

    "campaign_phases": [
        {{
            "phase": 1,
            "name": "Immediate Outreach",
            "timing": "Now",
            "channel": "phone|email|sms|in_app",
            "action": "What to do",
            "script_summary": "Key talking points",
            "success_criteria": "How to know it worked"
        }},
        {{
            "phase": 2,
            "name": "Follow-up",
            "timing": "If phase 1 fails within 48h",
            "channel": "channel",
            "action": "Action",
            "script_summary": "Key points",
            "success_criteria": "Criteria"
        }}
    ],

    "offers_sequence": [
        {{
            "order": 1,
            "offer_type": "discount|upgrade|credit|bundle",
            "offer_value": 100.00,
            "offer_description": "20% off next 3 months",
            "conditions": "12 month commitment",
            "cost_to_business": 50.00
        }}
    ],

    "escalation_path": {{
        "trigger": "When to escalate (e.g., 'customer threatens cancellation')",
        "escalate_to": "Role to escalate to",
        "authority_limits": "What escalation contact can offer",
        "script_for_handoff": "How to position the handoff"
    }},

    "negotiation_limits": {{
        "max_discount_percent": 30,
        "max_credit_amount": 200.00,
        "contract_flexibility": "Can waive early termination",
        "product_changes": "Can offer temporary upgrade"
    }},

    "win_back_plan": {{
        "wait_period_days": 30,
        "trigger_conditions": ["Competitor issues", "Time-based outreach"],
        "win_back_offer": "Best offer for return",
        "approach": "How to reach out"
    }},

    "strategy_rationale": "Why this strategy fits this customer",
    "risk_factors": ["What could go wrong"],
    "success_probability": 0.X
}}"""

        urgency_level = plan.get("urgency_level", "moderate")

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
                "Retention strategy response JSON parsing failed, using fallback",
                error=str(e),
                customer_id=context.customer_id
            )
            return self._get_fallback_strategy(context, urgency_level, f"JSON parse error: {e}")

        except Exception as e:
            logger.error("Retention strategy generation failed", error=str(e))
            return self._get_fallback_strategy(context, urgency_level, str(e))

    def _get_fallback_strategy(
        self,
        context: SubAgentContext,
        urgency: str,
        error: str
    ) -> Dict[str, Any]:
        """Return fallback retention strategy when generation fails."""
        return {
            "strategy_type": "reactive",
            "urgency": urgency,
            "estimated_retention_probability": 0.5,
            "campaign_phases": [],
            "offers_sequence": [],
            "escalation_path": {},
            "negotiation_limits": {},
            "win_back_plan": {},
            "strategy_rationale": f"Strategy generation failed: {error}",
            "error": error,
            "customer_id": context.customer_id
        }
