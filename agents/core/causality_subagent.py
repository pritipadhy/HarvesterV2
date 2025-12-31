"""
Causality Sub-Agent for Harvester V2
=====================================

Performs causal reasoning on customer data:
- Build causal DAG (directed acyclic graph)
- Root cause analysis
- What-if counterfactual simulation
- Harvest priority recommendations

The causality sub-agent provides deep reasoning about WHY things happen,
not just correlations but actual causal relationships.

Usage:
    subagent = CausalitySubAgent()
    result = await subagent.run_isolated(
        context,
        "Analyze causal relationships"
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
class CausalitySubAgent(BaseSubAgent):
    """
    Causality sub-agent for causal reasoning.

    Outputs:
    1. causal_dag: Directed graph of causal relationships
    2. root_cause: Primary driver of the target outcome
    3. what_if: Counterfactual simulation results
    4. harvest_priorities: What data to collect next
    """

    agent_name = "causality_subagent"

    def __init__(self, options: Dict[str, Any] = None):
        super().__init__(agent_name="causality_subagent", options=options or {})

    async def _plan(self, context: SubAgentContext, task: str) -> Dict[str, Any]:
        """Plan causal analysis approach."""
        # Identify target outcome from input
        target_outcome = context.input_data.get("target_outcome", "churn")

        # Get available variables from input data
        available_vars = list(context.input_data.keys())

        return {
            "analysis_types": ["root_cause", "causal_dag", "what_if"],
            "target_outcome": target_outcome,
            "candidate_causes": available_vars,
            "use_scores": "scores" in context.input_data
        }

    async def _execute_plan(self, context: SubAgentContext, plan: Dict) -> Dict[str, Any]:
        """Execute causal analysis."""
        prompt = f"""You are a causal inference expert analyzing customer behavior.

CUSTOMER DATA:
{json.dumps(context.input_data, indent=2, default=str)}

ANALYSIS PLAN:
{json.dumps(plan, indent=2, default=str)}

TARGET OUTCOME: {plan.get('target_outcome', 'churn')}

Perform comprehensive causal analysis:

1. CAUSAL DAG: Identify causal relationships between variables
   - What causes what?
   - What are the intermediate effects?
   - What are confounders?

2. ROOT CAUSE: What is the primary driver of the target outcome?
   - Not just correlation, but actual causation
   - Consider temporal ordering
   - Consider intervention points

3. WHAT-IF: If we intervene on the root cause, what happens?
   - Counterfactual reasoning
   - Expected impact magnitude
   - Confidence in prediction

4. HARVEST PRIORITIES: What data gaps need filling?
   - What would strengthen our causal understanding?
   - What confounders are we missing?

Respond with ONLY a JSON object (no markdown):
{{
    "causal_dag": {{
        "nodes": ["var1", "var2", "outcome"],
        "edges": [
            {{"from": "var1", "to": "outcome", "strength": 0.X, "type": "direct|indirect"}},
            {{"from": "var2", "to": "var1", "strength": 0.X, "type": "direct|indirect"}}
        ],
        "confounders": ["confounder1"]
    }},
    "root_cause": {{
        "variable": "primary driver variable",
        "causal_effect": 0.X,
        "confidence": 0.X,
        "explanation": "Why this is the root cause",
        "temporal_evidence": "Evidence of temporal ordering"
    }},
    "what_if": {{
        "intervention": "Description of intervention on root cause",
        "target_variable": "root cause variable",
        "intervention_value": "new value or change",
        "expected_outcome_change": 0.X,
        "confidence": 0.X,
        "reasoning": "Why this intervention would work",
        "side_effects": ["potential side effect 1"]
    }},
    "harvest_priorities": [
        {{"field": "data field needed", "priority": "urgent|high|medium", "reason": "why needed for causal analysis"}}
    ],
    "causal_summary": "One sentence summary of causal findings"
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
                "Causality response JSON parsing failed, using fallback",
                error=str(e),
                customer_id=context.customer_id
            )
            return self._get_fallback_causality(context, f"JSON parse error: {e}")

        except Exception as e:
            logger.error("Causal analysis failed", error=str(e))
            return self._get_fallback_causality(context, str(e))

    def _get_fallback_causality(self, context: SubAgentContext, error: str) -> Dict[str, Any]:
        """Return fallback causality analysis when generation fails."""
        return {
            "causal_dag": {"nodes": [], "edges": [], "confounders": []},
            "root_cause": {"variable": "unknown", "causal_effect": 0, "confidence": 0},
            "what_if": {"intervention": "unknown", "expected_outcome_change": 0},
            "harvest_priorities": [],
            "error": error,
            "customer_id": context.customer_id
        }
