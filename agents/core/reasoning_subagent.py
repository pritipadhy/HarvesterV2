"""
Reasoning Sub-Agent for Harvester V2
=====================================

Deep strategic reasoning for long-term insights:
- Pattern detection across customer base
- Future predictions (30/60/90 days)
- Data gap identification
- Proactive harvesting recommendations
- Playbook updates

This replaces Module 29's 5-day cycle with on-demand deep reasoning
that runs as part of the intelligence workflow.

Usage:
    subagent = ReasoningSubAgent()
    result = await subagent.run_isolated(
        context,
        "Generate strategic insights"
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
class ReasoningSubAgent(BaseSubAgent):
    """
    Deep reasoning sub-agent for strategic insights.

    Higher quality bar (0.90 critique threshold) for strategic reasoning.

    Outputs:
    1. patterns_detected: Patterns in the data
    2. predictions: Forward-looking predictions
    3. future_data_needs: Data gaps to fill
    4. proactive_harvesting: Entities to harvest proactively
    5. strategic_guidance: Long-term recommendations
    6. playbooks_to_update: Playbook improvement suggestions
    """

    agent_name = "reasoning_subagent"

    # Higher bar for strategic reasoning
    critique_threshold = 0.90

    def __init__(self, options: Dict[str, Any] = None):
        super().__init__(agent_name="reasoning_subagent", options=options or {})

    async def _plan(self, context: SubAgentContext, task: str) -> Dict[str, Any]:
        """Plan deep reasoning analysis."""
        return {
            "analysis_horizons": ["immediate", "short_term", "long_term"],
            "lookback_days": context.input_data.get("lookback_days", 90),
            "focus_areas": ["patterns", "predictions", "data_gaps", "proactive_actions"],
            "has_historical": bool(context.memory_snapshot),
            "has_scores": "scores" in context.input_data
        }

    async def _execute_plan(self, context: SubAgentContext, plan: Dict) -> Dict[str, Any]:
        """Execute deep reasoning analysis."""
        prompt = f"""You are performing deep strategic reasoning on customer intelligence.

CONTEXT:
Customer ID: {context.customer_id}
Tenant: {context.tenant_id}

INPUT DATA:
{json.dumps(context.input_data, indent=2, default=str)}

ANALYSIS PLAN:
{json.dumps(plan, indent=2, default=str)}

HISTORICAL MEMORY:
{json.dumps(context.memory_snapshot, indent=2, default=str) if context.memory_snapshot else "(none available)"}

Perform comprehensive strategic analysis:

1. PATTERN DETECTION: What patterns emerge from the data?
   - Behavioral patterns
   - Usage patterns
   - Engagement patterns
   - Risk patterns

2. PREDICTIONS: What will happen in the next 30/60/90 days?
   - Based on current trajectory
   - Based on similar customer patterns
   - Confidence levels for each

3. DATA GAPS: What data would improve our predictions?
   - Missing signals
   - Stale data
   - Confounders we can't measure

4. PROACTIVE ACTIONS: What should we do before problems occur?
   - Early interventions
   - Relationship building
   - Risk mitigation

5. PLAYBOOK OPPORTUNITIES: What success patterns should we codify?
   - What worked for similar customers?
   - What scripts/actions had best outcomes?

Respond with ONLY a JSON object (no markdown):
{{
    "patterns_detected": [
        {{"pattern": "Description", "confidence": 0.X, "impact": "high|medium|low", "evidence": ["e1", "e2"]}}
    ],
    "predictions": [
        {{
            "prediction": "What will happen",
            "timeframe_days": 30,
            "confidence": 0.X,
            "supporting_evidence": ["evidence1"],
            "risk_if_ignored": "What happens if we don't act"
        }}
    ],
    "future_data_needs": [
        {{"field": "Data field needed", "priority": "urgent|high|medium", "reason": "Why this data matters"}}
    ],
    "proactive_harvesting": [
        {{"entity_type": "Customer|Partner|Employee", "entity_id": "ID if known", "reason": "Why harvest proactively"}}
    ],
    "strategic_guidance": [
        {{"recommendation": "Strategic recommendation", "impact": "Expected impact", "effort": "high|medium|low"}}
    ],
    "playbooks_to_update": [
        {{"playbook_type": "retention|upsell|service", "update": "Suggested update", "reason": "Why update"}}
    ],
    "reasoning_summary": "Executive summary of key insights"
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
                "Reasoning response JSON parsing failed, using fallback",
                error=str(e),
                customer_id=context.customer_id
            )
            return self._get_fallback_reasoning(context, f"JSON parse error: {e}")

        except Exception as e:
            logger.error("Deep reasoning failed", error=str(e))
            return self._get_fallback_reasoning(context, str(e))

    def _get_fallback_reasoning(self, context: SubAgentContext, error: str) -> Dict[str, Any]:
        """Return fallback reasoning when generation fails."""
        return {
            "patterns_detected": [],
            "predictions": [],
            "future_data_needs": [],
            "proactive_harvesting": [],
            "strategic_guidance": [],
            "playbooks_to_update": [],
            "reasoning_summary": f"Analysis failed: {error}",
            "error": error,
            "customer_id": context.customer_id
        }
