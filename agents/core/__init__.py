"""
Harvester v2 Core Sub-Agents - THE ULTIMATE SYSTEM

Production-level sub-agents for customer intelligence:

Sub-agents:
    - ScoringSubAgent: Fraud, churn, propensity, risk scoring
    - CausalitySubAgent: Causal DAG, root cause, what-if analysis
    - NBASubAgent: Next best action generation with personalized scripts
    - ReasoningSubAgent: Deep strategic reasoning (replaces Module 29)
    - RetainerStrategySubAgent: Retention negotiation strategies
    - ResearchSubAgent: Web search and synthesis with reflection loop
    - OpportunitySubAgent: Opportunity detection (upsell, cross-sell, expansion)
    - RiskSignalSubAgent: Risk signal aggregation (churn, fraud, competitive)

THE ULTIMATE SYSTEM combines:
    - Deep research (EnterpriseGPT-style with reflection loop)
    - Opportunity detection (finds growth signals)
    - Risk signal aggregation (detects threats)
    - Actionable recommendations (NBA, retention strategies)

Each sub-agent follows the Plan -> Execute -> Critique pattern with
isolated contexts and automatic Langfuse tracing.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harvester_v2.agents.core.scoring_subagent import ScoringSubAgent
    from harvester_v2.agents.core.causality_subagent import CausalitySubAgent
    from harvester_v2.agents.core.nba_subagent import NBASubAgent
    from harvester_v2.agents.core.reasoning_subagent import ReasoningSubAgent
    from harvester_v2.agents.core.retainer_strategy_subagent import RetainerStrategySubAgent
    from harvester_v2.agents.core.research_subagent import ResearchSubAgent
    from harvester_v2.agents.core.opportunity_subagent import OpportunitySubAgent
    from harvester_v2.agents.core.risk_signal_subagent import RiskSignalSubAgent

__all__ = [
    "ScoringSubAgent",
    "CausalitySubAgent",
    "NBASubAgent",
    "ReasoningSubAgent",
    "RetainerStrategySubAgent",
    "ResearchSubAgent",
    "OpportunitySubAgent",
    "RiskSignalSubAgent",
]
