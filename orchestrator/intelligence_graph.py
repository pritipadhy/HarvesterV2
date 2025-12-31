"""
LangGraph-based Intelligence Orchestration for Harvester V2
============================================================

Uses LangGraph StateGraph for workflow orchestration with:
- Durable checkpointing via AsyncPostgresSaver
- Langfuse tracing integration
- Phased sub-agent execution
- Quality critique and retry logic

Workflow stages:
1. INITIALIZE - Load memory, validate input
2. PHASE_1_SCORING - Run scoring sub-agent
3. PHASE_2_ANALYSIS - Run causality + reasoning in parallel
4. PHASE_3_ACTIONS - Run NBA + retention in parallel
5. AGGREGATE - Combine results with LLM reasoning
6. CRITIQUE - Evaluate quality (retry if < 0.85)
7. COMPLETE - Finalize and emit

Usage:
    graph = HarvesterV2Graph(tenant_id="gigaclear")
    result = await graph.run({
        "customer_id": "CUST-123",
        "tenant_id": "gigaclear",
        "context": {"product": "broadband", "tenure_months": 24}
    })
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from datetime import datetime
import uuid
import json
import asyncio
import structlog

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    AsyncPostgresSaver = None

from harvester_v2.agents.base_subagent import SubAgentContext, SubAgentResult
from harvester_v2.agents.parallel_executor import ParallelSubAgentExecutor, SubAgentRegistry
from shared.ai.model_router import get_model_router, TaskType
from shared.logging import get_platform_logger

# Try to import Langfuse tracer
try:
    from shared.observability import get_langfuse_tracer
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    get_langfuse_tracer = None

# Try to import LangGraph adapter
try:
    from shared.state_management.langgraph_adapter import get_async_langgraph_adapter
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False
    get_async_langgraph_adapter = None

# Try to import context compaction
try:
    from shared.context_compaction.compactor import get_context_compactor
    from shared.context_compaction.adaptive_windows import get_window_manager
    CONTEXT_COMPACTION_AVAILABLE = True
except ImportError:
    CONTEXT_COMPACTION_AVAILABLE = False
    get_context_compactor = None
    get_window_manager = None

# Try to import evaluation framework
try:
    from shared.evaluation.hallucination import get_hallucination_detector
    from shared.evaluation.evaluator import get_agent_evaluator
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    get_hallucination_detector = None
    get_agent_evaluator = None

logger = get_platform_logger(__name__)


class HarvesterV2State(TypedDict, total=False):
    """
    LangGraph state for Harvester V2 workflow.

    This TypedDict defines all state that flows through the graph.
    State is immutable - each node returns a new state dict.

    THE ULTIMATE SYSTEM - combines:
    - Deep research (EnterpriseGPT-style with reflection loop)
    - Opportunity detection (upsell, cross-sell signals)
    - Risk signal aggregation (churn, fraud, competitive)
    - Actionable recommendations (NBA, retention strategies)
    """
    # Core identifiers
    workflow_id: str
    customer_id: str
    tenant_id: str
    session_id: str

    # Langfuse integration (propagate trace)
    _langfuse_trace_id: Optional[str]

    # Input data
    input_context: Dict[str, Any]
    memory_snapshot: Dict[str, Any]

    # Context squeezing fields
    squeezed_memory_snapshot: Optional[str]
    token_budget: int
    squeeze_metadata: Dict[str, Any]

    # Sub-agent results (populated by each phase)
    scoring_result: Optional[Dict[str, Any]]
    causality_result: Optional[Dict[str, Any]]
    reasoning_result: Optional[Dict[str, Any]]
    research_result: Optional[Dict[str, Any]]  # Web search and synthesis with reflection
    opportunity_result: Optional[Dict[str, Any]]  # NEW: Opportunity detection
    risk_signal_result: Optional[Dict[str, Any]]  # NEW: Risk signal aggregation
    nba_result: Optional[Dict[str, Any]]
    retention_result: Optional[Dict[str, Any]]

    # Aggregated output
    final_intelligence: Optional[Dict[str, Any]]

    # Workflow tracking
    current_stage: str
    stage_history: List[str]
    errors: List[Dict[str, Any]]

    # Critique and evaluation
    critique_score: float
    iteration_count: int
    evaluation_result: Optional[Dict[str, Any]]
    hallucination_warnings: List[Dict[str, Any]]


class HarvesterV2Graph:
    """
    LangGraph-based orchestration for Harvester V2.

    This is the main entry point for the intelligence engine.
    It orchestrates sub-agents through a phased workflow with
    automatic checkpointing and quality control.

    Phases:
    1. Scoring (independent)
    2. Causality + Reasoning (parallel, uses scoring)
    3. NBA + Retention (parallel, uses all above)

    Quality control:
    - Each phase is critiqued
    - Retry up to 3 times if quality < 0.85
    - Final aggregation synthesizes all results
    """

    # Quality threshold for accepting results
    critique_threshold: float = 0.85

    # Maximum iterations for quality improvement
    max_iterations: int = 3

    def __init__(
        self,
        tenant_id: str,
        use_checkpointing: bool = True,
        use_langfuse: bool = True
    ):
        """
        Initialize the intelligence graph.

        Args:
            tenant_id: Tenant identifier
            use_checkpointing: Enable durable checkpointing
            use_langfuse: Enable Langfuse tracing
        """
        self.tenant_id = tenant_id
        self.use_checkpointing = use_checkpointing
        self.use_langfuse = use_langfuse and LANGFUSE_AVAILABLE

        # Initialize model router
        self.model_router = get_model_router()

        # Initialize Langfuse tracer if available
        self.tracer = None
        if self.use_langfuse and get_langfuse_tracer:
            try:
                self.tracer = get_langfuse_tracer()
            except Exception as e:
                logger.warning("Langfuse tracer not available", error=str(e))

        # Load sub-agents for this tenant
        self._load_subagents()

        # Build the graph
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

        logger.info(
            "HarvesterV2Graph initialized",
            tenant_id=tenant_id,
            use_checkpointing=use_checkpointing,
            use_langfuse=self.use_langfuse,
            langgraph_available=LANGGRAPH_AVAILABLE
        )

    def _load_subagents(self):
        """Load sub-agents for the tenant (THE ULTIMATE SYSTEM)."""
        # Import sub-agents here to avoid circular imports
        try:
            from harvester_v2.agents.core.scoring_subagent import ScoringSubAgent
            from harvester_v2.agents.core.causality_subagent import CausalitySubAgent
            from harvester_v2.agents.core.reasoning_subagent import ReasoningSubAgent
            from harvester_v2.agents.core.research_subagent import ResearchSubAgent
            from harvester_v2.agents.core.opportunity_subagent import OpportunitySubAgent
            from harvester_v2.agents.core.risk_signal_subagent import RiskSignalSubAgent
            from harvester_v2.agents.core.nba_subagent import NBASubAgent
            from harvester_v2.agents.core.retainer_strategy_subagent import RetainerStrategySubAgent

            self.scoring = ScoringSubAgent()
            self.causality = CausalitySubAgent()
            self.reasoning = ReasoningSubAgent()
            self.research = ResearchSubAgent()  # Web search with reflection loop
            self.opportunity = OpportunitySubAgent()  # NEW: Opportunity detection
            self.risk_signal = RiskSignalSubAgent()  # NEW: Risk signal aggregation
            self.nba = NBASubAgent()
            self.retention = RetainerStrategySubAgent()

            # Create executor with all sub-agents
            self.executor = ParallelSubAgentExecutor([
                self.scoring,
                self.causality,
                self.reasoning,
                self.research,
                self.opportunity,
                self.risk_signal,
                self.nba,
                self.retention
            ])

            logger.info(
                "Sub-agents loaded (THE ULTIMATE SYSTEM)",
                agents=self.executor.get_agent_names(),
                capabilities=["deep_research", "opportunity_detection", "risk_signals", "actions"]
            )

        except ImportError as e:
            logger.warning(
                "Sub-agent import failed - running in simplified mode without parallel execution",
                error=str(e),
                hint="Ensure all sub-agent dependencies are installed"
            )
            self.executor = None

    def _build_graph(self) -> Optional[StateGraph]:
        """Build the LangGraph workflow."""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available, graph not built")
            return None

        graph = StateGraph(HarvesterV2State)

        # Add nodes for each stage
        graph.add_node("initialize", self._initialize_node)
        graph.add_node("phase_1_scoring", self._scoring_node)
        graph.add_node("phase_2_analysis", self._analysis_node)
        graph.add_node("phase_3_actions", self._actions_node)
        graph.add_node("aggregate", self._aggregate_node)
        graph.add_node("critique", self._critique_node)
        graph.add_node("complete", self._complete_node)

        # Set entry point
        graph.set_entry_point("initialize")

        # Add edges for linear flow
        graph.add_edge("initialize", "phase_1_scoring")
        graph.add_edge("phase_1_scoring", "phase_2_analysis")
        graph.add_edge("phase_2_analysis", "phase_3_actions")
        graph.add_edge("phase_3_actions", "aggregate")
        graph.add_edge("aggregate", "critique")

        # Conditional edge: retry or complete
        graph.add_conditional_edges(
            "critique",
            self._should_retry,
            {
                "retry": "phase_1_scoring",
                "complete": "complete"
            }
        )

        graph.add_edge("complete", END)

        return graph

    # =========================================================================
    # Graph Nodes
    # =========================================================================

    async def _initialize_node(self, state: HarvesterV2State) -> Dict[str, Any]:
        """
        Initialize workflow with memory loading and context squeezing.

        Loads relevant episodic and semantic memory for the customer,
        then compresses it for token efficiency.
        """
        logger.info(
            "Initializing workflow",
            workflow_id=state.get("workflow_id"),
            customer_id=state.get("customer_id")
        )

        # Load memory
        memory_snapshot = {}
        memories_list = []
        try:
            from harvester_v2.memory.episodic_memory import EpisodicMemory
            memory = EpisodicMemory(state["tenant_id"])
            memory_snapshot = await memory.get_for_entity(state["customer_id"])
            memories_list = memory_snapshot.get("memories", [])
        except ImportError as e:
            logger.debug("Memory module import failed", error=str(e))
        except Exception as e:
            logger.warning("Failed to load memory", error=str(e))

        # Context squeezing - compress memories if they exceed token budget
        squeezed_memory_snapshot = None
        squeeze_metadata = {}
        token_budget = 40000  # Default budget

        if CONTEXT_COMPACTION_AVAILABLE and get_context_compactor and memories_list:
            try:
                compactor = get_context_compactor()
                window_manager = get_window_manager()

                # Get adaptive token budget for harvester workflow
                config = window_manager.get_config(
                    agent_name="harvester_intelligence",
                    tier="professional"
                )
                token_budget = config.get("target_tokens", 40000)

                # Convert memories to message format for compaction
                memory_messages = [
                    {
                        "role": "system",
                        "content": f"Memory [{m.get('memory_type', 'general')}]: {m.get('content', '')}",
                        "timestamp": m.get("timestamp", "")
                    }
                    for m in memories_list if isinstance(m, dict)
                ]

                # Check if compaction is needed
                if memory_messages and compactor.needs_compaction(
                    memory_messages,
                    agent_name="harvester_intelligence"
                ):
                    compacted = await compactor.compact(
                        messages=memory_messages,
                        agent_name="harvester_intelligence",
                        session_id=state.get("session_id"),
                        user_id=state.get("customer_id")
                    )

                    squeezed_memory_snapshot = compacted.get("summary")
                    squeeze_metadata = {
                        "original_count": len(memories_list),
                        "tokens_before": compacted.get("original_tokens", 0),
                        "tokens_after": compacted.get("compacted_tokens", 0),
                        "compression_ratio": compacted.get("reduction_percent", 0) / 100,
                        "compaction_performed": True
                    }

                    logger.info(
                        "Memory context squeezed",
                        workflow_id=state.get("workflow_id"),
                        original_memories=len(memories_list),
                        compression_ratio=f"{squeeze_metadata['compression_ratio']:.0%}"
                    )
                else:
                    # No compaction needed - create simple summary
                    squeeze_metadata = {
                        "original_count": len(memories_list),
                        "compaction_performed": False
                    }
            except Exception as e:
                logger.warning("Context squeezing failed, using raw memories", error=str(e))

        return {
            "memory_snapshot": memory_snapshot,
            "squeezed_memory_snapshot": squeezed_memory_snapshot,
            "token_budget": token_budget,
            "squeeze_metadata": squeeze_metadata,
            "current_stage": "initialized",
            "stage_history": [*state.get("stage_history", []), "initialized"]
        }

    async def _scoring_node(self, state: HarvesterV2State) -> Dict[str, Any]:
        """
        Run scoring sub-agent.

        Calculates fraud, churn, propensity, and risk scores.
        """
        logger.info(
            "Phase 1: Scoring",
            workflow_id=state.get("workflow_id"),
            customer_id=state.get("customer_id")
        )

        result_data = None

        if self.executor and hasattr(self, 'scoring'):
            context = SubAgentContext(
                tenant_id=state["tenant_id"],
                customer_id=state.get("customer_id"),
                input_data=state.get("input_context", {}),
                memory_snapshot=state.get("memory_snapshot", {}),
                orchestrator_run_id=state.get("workflow_id"),
                langfuse_trace_id=state.get("_langfuse_trace_id"),
                # Context squeezing fields
                squeezed_memory_snapshot=state.get("squeezed_memory_snapshot"),
                token_budget=state.get("token_budget", 40000),
                squeeze_metadata=state.get("squeeze_metadata", {})
            )

            result = await self.scoring.run_isolated(
                context,
                "Calculate customer risk scores"
            )

            if result.success:
                result_data = result.data
            else:
                logger.warning(
                    "Scoring sub-agent failed",
                    error=result.error
                )

        return {
            "scoring_result": result_data,
            "current_stage": "phase_1_complete",
            "stage_history": [*state.get("stage_history", []), "phase_1_complete"]
        }

    async def _analysis_node(self, state: HarvesterV2State) -> Dict[str, Any]:
        """
        Run analysis sub-agents in parallel (THE ULTIMATE SYSTEM).

        Phase 2 runs:
        - Causality analysis (root cause, what-if)
        - Strategic reasoning (patterns, predictions)
        - Deep research (web search with reflection loop)
        - Opportunity detection (upsell, cross-sell signals)
        - Risk signal aggregation (churn, fraud, competitive)

        Uses scoring results from Phase 1.
        """
        logger.info(
            "Phase 2: Analysis (Causality + Reasoning + Research + Opportunity + Risk)",
            workflow_id=state.get("workflow_id")
        )

        causality_result = None
        reasoning_result = None
        research_result = None
        opportunity_result = None
        risk_signal_result = None

        if self.executor and hasattr(self, 'causality') and hasattr(self, 'reasoning'):
            # Build context with scoring results
            scoring_result = state.get("scoring_result", {})
            input_data = {
                **state.get("input_context", {}),
                "scores": scoring_result,
                # Pass scoring results directly for opportunity/risk analysis
                "churn_risk": scoring_result.get("churn_risk"),
                "fraud_score": scoring_result.get("fraud_score"),
                "propensity_score": scoring_result.get("propensity_score"),
                "risk_score": scoring_result.get("risk_score"),
                "customer_segment": scoring_result.get("customer_segment")
            }

            context = SubAgentContext(
                tenant_id=state["tenant_id"],
                customer_id=state.get("customer_id"),
                input_data=input_data,
                memory_snapshot=state.get("memory_snapshot", {}),
                orchestrator_run_id=state.get("workflow_id"),
                langfuse_trace_id=state.get("_langfuse_trace_id"),
                # Context squeezing fields
                squeezed_memory_snapshot=state.get("squeezed_memory_snapshot"),
                token_budget=state.get("token_budget", 40000),
                squeeze_metadata=state.get("squeeze_metadata", {})
            )

            # Create tasks for all Phase 2 sub-agents
            tasks = []
            task_names = []

            # Core analysis
            causality_task = self.causality.run_isolated(
                context.copy(),
                "Analyze causal relationships"
            )
            tasks.append(causality_task)
            task_names.append("causality")

            reasoning_task = self.reasoning.run_isolated(
                context.copy(),
                "Generate strategic insights"
            )
            tasks.append(reasoning_task)
            task_names.append("reasoning")

            # Deep research with reflection loop
            if hasattr(self, 'research'):
                research_task = self.research.run_isolated(
                    context.copy(),
                    "Research market context and competitor insights"
                )
                tasks.append(research_task)
                task_names.append("research")

            # NEW: Opportunity detection
            if hasattr(self, 'opportunity'):
                opportunity_task = self.opportunity.run_isolated(
                    context.copy(),
                    "Identify opportunities for customer growth"
                )
                tasks.append(opportunity_task)
                task_names.append("opportunity")

            # NEW: Risk signal aggregation
            if hasattr(self, 'risk_signal'):
                risk_signal_task = self.risk_signal.run_isolated(
                    context.copy(),
                    "Aggregate risk signals for customer"
                )
                tasks.append(risk_signal_task)
                task_names.append("risk_signal")

            # Run all in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results by name
            for i, (name, result) in enumerate(zip(task_names, results)):
                if isinstance(result, Exception):
                    logger.warning(f"{name} sub-agent failed", error=str(result))
                    continue

                if not result.success:
                    logger.warning(f"{name} sub-agent failed", error=result.error)
                    continue

                if name == "causality":
                    causality_result = result.data
                elif name == "reasoning":
                    reasoning_result = result.data
                elif name == "research":
                    research_result = result.data
                    logger.info(
                        "Research sub-agent completed (with reflection loop)",
                        urls_found=len(research_result.get("source_urls", [])),
                        reflection_iterations=research_result.get("reflection_iterations", 0)
                    )
                elif name == "opportunity":
                    opportunity_result = result.data
                    logger.info(
                        "Opportunity sub-agent completed",
                        opportunities_found=len(opportunity_result.get("opportunities", []))
                    )
                elif name == "risk_signal":
                    risk_signal_result = result.data
                    logger.info(
                        "Risk signal sub-agent completed",
                        risk_signals_found=len(risk_signal_result.get("risk_signals", []))
                    )

        return {
            "causality_result": causality_result,
            "reasoning_result": reasoning_result,
            "research_result": research_result,
            "opportunity_result": opportunity_result,
            "risk_signal_result": risk_signal_result,
            "current_stage": "phase_2_complete",
            "stage_history": [*state.get("stage_history", []), "phase_2_complete"]
        }

    async def _actions_node(self, state: HarvesterV2State) -> Dict[str, Any]:
        """
        Run NBA and retention sub-agents in parallel.

        Uses all previous results.
        """
        logger.info(
            "Phase 3: Actions (NBA + Retention)",
            workflow_id=state.get("workflow_id")
        )

        nba_result = None
        retention_result = None

        if self.executor and hasattr(self, 'nba') and hasattr(self, 'retention'):
            # Build context with all previous results
            input_data = {
                **state.get("input_context", {}),
                "scores": state.get("scoring_result", {}),
                "causal_analysis": state.get("causality_result", {}),
                "strategic_insights": state.get("reasoning_result", {})
            }

            context = SubAgentContext(
                tenant_id=state["tenant_id"],
                customer_id=state.get("customer_id"),
                input_data=input_data,
                memory_snapshot=state.get("memory_snapshot", {}),
                orchestrator_run_id=state.get("workflow_id"),
                langfuse_trace_id=state.get("_langfuse_trace_id"),
                # Context squeezing fields
                squeezed_memory_snapshot=state.get("squeezed_memory_snapshot"),
                token_budget=state.get("token_budget", 40000),
                squeeze_metadata=state.get("squeeze_metadata", {})
            )

            # Run both in parallel
            nba_task = self.nba.run_isolated(
                context.copy(),
                "Generate next best action"
            )
            retention_task = self.retention.run_isolated(
                context.copy(),
                "Generate retention strategy"
            )

            results = await asyncio.gather(
                nba_task,
                retention_task,
                return_exceptions=True
            )

            # Process results
            if not isinstance(results[0], Exception) and results[0].success:
                nba_result = results[0].data
            else:
                logger.warning("NBA sub-agent failed", error=str(results[0]))

            if not isinstance(results[1], Exception) and results[1].success:
                retention_result = results[1].data
            else:
                logger.warning("Retention sub-agent failed", error=str(results[1]))

        return {
            "nba_result": nba_result,
            "retention_result": retention_result,
            "current_stage": "phase_3_complete",
            "stage_history": [*state.get("stage_history", []), "phase_3_complete"]
        }

    async def _aggregate_node(self, state: HarvesterV2State) -> Dict[str, Any]:
        """
        Aggregate all sub-agent results with LLM reasoning (THE ULTIMATE SYSTEM).

        Synthesizes results into a coherent intelligence package that includes:
        - Scores and risk assessment
        - Causal analysis
        - Deep research findings
        - Opportunities identified
        - Risk signals detected
        - Next best actions
        - Retention strategy
        """
        logger.info(
            "Aggregating results (THE ULTIMATE SYSTEM)",
            workflow_id=state.get("workflow_id")
        )

        # Include research if available
        research_result = state.get('research_result', {})
        research_section = ""
        if research_result and research_result.get("source_urls"):
            research_section = f"""
RESEARCH (Deep Web Search with Reflection):
{json.dumps(research_result, indent=2, default=str)[:3000]}
"""

        # Include opportunity if available
        opportunity_result = state.get('opportunity_result', {})
        opportunity_section = ""
        if opportunity_result and opportunity_result.get("opportunities"):
            opportunity_section = f"""
OPPORTUNITIES (Detected Growth Potential):
{json.dumps(opportunity_result, indent=2, default=str)[:2000]}
"""

        # Include risk signals if available
        risk_signal_result = state.get('risk_signal_result', {})
        risk_signal_section = ""
        if risk_signal_result and risk_signal_result.get("risk_signals"):
            risk_signal_section = f"""
RISK SIGNALS (Aggregated Threats):
{json.dumps(risk_signal_result, indent=2, default=str)[:2000]}
"""

        prompt = f"""You are aggregating intelligence from THE ULTIMATE SYSTEM - combining deep research,
opportunity detection, and risk signal aggregation into actionable customer intelligence.

CUSTOMER: {state.get('customer_id')}
TENANT: {state.get('tenant_id')}

SUB-AGENT RESULTS:

SCORING:
{json.dumps(state.get('scoring_result', {}), indent=2, default=str)}

CAUSALITY:
{json.dumps(state.get('causality_result', {}), indent=2, default=str)}

REASONING:
{json.dumps(state.get('reasoning_result', {}), indent=2, default=str)}
{research_section}
{opportunity_section}
{risk_signal_section}
NBA:
{json.dumps(state.get('nba_result', {}), indent=2, default=str)}

RETENTION:
{json.dumps(state.get('retention_result', {}), indent=2, default=str)}

Synthesize ALL results into a unified intelligence summary.
This is THE ULTIMATE SYSTEM - integrate:
- Deep research findings (with reflection loop)
- Detected opportunities (upsell, cross-sell)
- Aggregated risk signals (churn, fraud, competitive)
- Actionable recommendations (NBA, retention strategy)

Respond with ONLY a JSON object (no markdown):
{{
    "scores": {{
        "fraud_score": 0.X,
        "churn_risk": 0.X,
        "propensity_score": 0.X,
        "overall_health": 0.X
    }},
    "causal_analysis": {{
        "root_cause": "primary driver",
        "causal_chain": ["cause1 -> effect1"],
        "intervention_recommended": "what to do"
    }},
    "opportunities": [
        {{"type": "upsell|cross_sell|expansion", "name": "...", "potential": "high|medium|low"}}
    ],
    "risk_signals": [
        {{"type": "churn|fraud|competitive", "severity": "critical|high|medium|low", "action_required": "..."}}
    ],
    "next_best_action": {{
        "action": "recommended action",
        "confidence": 0.X,
        "urgency": "immediate|24h|7d",
        "script": "talk track"
    }},
    "strategic_insights": {{
        "patterns": ["pattern1"],
        "predictions": ["prediction1"],
        "data_gaps": ["gap1"]
    }},
    "web_research": {{
        "search_queries": ["query1", "query2"],
        "source_urls": ["url1", "url2"],
        "synthesis": "Key findings from web research..."
    }},
    "reasoning_trace": "Brief explanation of conclusions"
}}"""

        try:
            response = await self.model_router.complete(
                prompt=prompt,
                task_type=TaskType.REASONING,
                agent_name="intelligence_aggregator",
                tenant_id=state["tenant_id"]
            )

            # Parse response
            response_text = response.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            final_intelligence = json.loads(response_text)

        except Exception as e:
            logger.warning("Aggregation parsing failed", error=str(e))
            # Include all results in fallback (THE ULTIMATE SYSTEM)
            research_result = state.get("research_result", {})
            opportunity_result = state.get("opportunity_result", {})
            risk_signal_result = state.get("risk_signal_result", {})
            final_intelligence = {
                "scores": state.get("scoring_result", {}),
                "causal_analysis": state.get("causality_result", {}),
                "opportunities": opportunity_result.get("opportunities", []),
                "risk_signals": risk_signal_result.get("risk_signals", []),
                "next_best_action": state.get("nba_result", {}),
                "strategic_insights": state.get("reasoning_result", {}),
                "web_research": {
                    "search_queries": research_result.get("search_queries", []),
                    "source_urls": research_result.get("source_urls", []),
                    "synthesis": research_result.get("research_synthesis", ""),
                    "reflection_iterations": research_result.get("reflection_iterations", 0)
                },
                "reasoning_trace": "Aggregation failed, raw results returned"
            }

        return {
            "final_intelligence": final_intelligence,
            "current_stage": "aggregated",
            "stage_history": [*state.get("stage_history", []), "aggregated"]
        }

    async def _critique_node(self, state: HarvesterV2State) -> Dict[str, Any]:
        """
        Critique the aggregated result with evaluation framework.

        Uses AgentEvaluator for trajectory scoring and hallucination detection.
        """
        logger.info(
            "Critiquing results",
            workflow_id=state.get("workflow_id"),
            iteration=state.get("iteration_count", 0) + 1
        )

        # Check if we have all required components
        has_scores = bool(state.get("scoring_result"))
        has_nba = bool(state.get("nba_result"))
        has_final = bool(state.get("final_intelligence"))

        # Base quality check
        if has_scores and has_nba and has_final:
            base_score = 0.90
        elif has_scores and has_final:
            base_score = 0.80
        elif has_final:
            base_score = 0.70
        else:
            base_score = 0.50

        # Enhanced evaluation using AgentEvaluator
        evaluation_result = {}
        final_score = base_score

        if EVALUATION_AVAILABLE and get_agent_evaluator and state.get("final_intelligence"):
            try:
                evaluator = get_agent_evaluator(
                    pass_threshold=self.critique_threshold,
                    enable_hallucination_check=True
                )

                # Build state history for trajectory evaluation
                state_history = [
                    {"stage": stage, "data": state.get(f"{stage.replace('_complete', '')}_result", {})}
                    for stage in state.get("stage_history", [])
                ]

                # Get final intelligence as result
                final_result = {
                    "response": json.dumps(state.get("final_intelligence", {})),
                    "status": "complete" if has_final else "incomplete"
                }

                # Build context for evaluation
                context = {
                    "input_context": state.get("input_context", {}),
                    "memory_snapshot": state.get("memory_snapshot", {}),
                    "squeezed_memory": state.get("squeezed_memory_snapshot"),
                    "available_tools": ["scoring", "causality", "reasoning", "nba", "retention"],
                    "data_retrieved": has_scores
                }

                # Run evaluation
                report = await evaluator.evaluate(
                    session_id=state.get("session_id", state.get("workflow_id", "")),
                    state_history=state_history,
                    final_result=final_result,
                    context=context
                )

                # Combine base score with evaluation score
                final_score = (base_score + report.overall_score) / 2

                evaluation_result = {
                    "trajectory_efficiency": report.trajectory_score.efficiency,
                    "trajectory_correctness": report.trajectory_score.correctness,
                    "trajectory_quality": report.trajectory_score.quality,
                    "hallucination_detected": report.hallucination_result.has_hallucination,
                    "hallucination_type": report.hallucination_result.hallucination_type,
                    "hallucination_severity": report.hallucination_result.severity,
                    "overall_score": report.overall_score,
                    "passed": report.passed,
                    "issues": report.issues,
                    "recommendations": report.recommendations
                }

                logger.info(
                    "Evaluation complete",
                    workflow_id=state.get("workflow_id"),
                    base_score=f"{base_score:.2f}",
                    eval_score=f"{report.overall_score:.2f}",
                    final_score=f"{final_score:.2f}",
                    hallucination_detected=report.hallucination_result.has_hallucination
                )

            except Exception as e:
                logger.warning("AgentEvaluator failed, using base score", error=str(e))
                final_score = base_score

        return {
            "critique_score": final_score,
            "evaluation_result": evaluation_result,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "current_stage": "critiqued",
            "stage_history": [*state.get("stage_history", []), "critiqued"]
        }

    def _should_retry(self, state: HarvesterV2State) -> Literal["retry", "complete"]:
        """
        Decide whether to retry or complete.

        Retries if:
        - Critique score below threshold
        - Not exceeded max iterations
        """
        score = state.get("critique_score", 0)
        iterations = state.get("iteration_count", 0)

        if score < self.critique_threshold and iterations < self.max_iterations:
            logger.info(
                "Retry decision: retry",
                score=score,
                threshold=self.critique_threshold,
                iteration=iterations
            )
            return "retry"

        logger.info(
            "Retry decision: complete",
            score=score,
            threshold=self.critique_threshold,
            iteration=iterations
        )
        return "complete"

    async def _complete_node(self, state: HarvesterV2State) -> Dict[str, Any]:
        """
        Finalize the workflow with hallucination verification.

        Performs final grounding check on recommendations and marks workflow complete.
        """
        logger.info(
            "Completing workflow",
            workflow_id=state.get("workflow_id"),
            final_score=state.get("critique_score")
        )

        # Verify recommendations for hallucinations
        hallucination_warnings = []

        if EVALUATION_AVAILABLE and get_hallucination_detector and state.get("final_intelligence"):
            try:
                detector = get_hallucination_detector()

                final_intel = state.get("final_intelligence", {})
                recommendations = final_intel.get("recommendations", [])

                # Build context for grounding verification
                context = {
                    "customer_data": state.get("input_context", {}),
                    "scores": state.get("scoring_result", {}),
                    "memory_snapshot": state.get("memory_snapshot", {}),
                    "data_retrieved": bool(state.get("scoring_result")),
                    "available_tools": ["scoring", "causality", "reasoning", "nba", "retention"]
                }

                # Check each recommendation for hallucination
                for rec in recommendations:
                    if isinstance(rec, dict):
                        reasoning = rec.get("reasoning", rec.get("rationale", ""))
                        action = rec.get("action", rec.get("recommendation", ""))

                        result = await detector.detect(
                            response=f"{action}: {reasoning}",
                            context=context,
                            state={"messages": [], "customer_type": "existing"}
                        )

                        if result.has_hallucination and result.confidence > 0.6:
                            warning = {
                                "action": action,
                                "hallucination_type": result.hallucination_type,
                                "severity": result.severity,
                                "confidence": result.confidence,
                                "evidence": result.evidence[:3],  # Limit to 3 pieces of evidence
                                "recommendation": result.recommendation
                            }
                            hallucination_warnings.append(warning)

                            logger.warning(
                                "Hallucination detected in recommendation",
                                workflow_id=state.get("workflow_id"),
                                action=action[:50],
                                type=result.hallucination_type,
                                severity=result.severity
                            )

                if hallucination_warnings:
                    logger.warning(
                        "Workflow complete with hallucination warnings",
                        workflow_id=state.get("workflow_id"),
                        warning_count=len(hallucination_warnings)
                    )

            except Exception as e:
                logger.debug("Hallucination verification failed", error=str(e))

        # Trace evaluation data if Langfuse is available
        if self.tracer and state.get("evaluation_result"):
            try:
                self.tracer.span(
                    name="evaluation_metrics",
                    metadata={
                        "critique_score": state.get("critique_score"),
                        "evaluation_result": state.get("evaluation_result"),
                        "squeeze_metadata": state.get("squeeze_metadata", {}),
                        "hallucination_warnings": len(hallucination_warnings)
                    }
                )
            except Exception as e:
                logger.debug("Failed to trace evaluation data", error=str(e))

        return {
            "hallucination_warnings": hallucination_warnings,
            "current_stage": "completed",
            "stage_history": [*state.get("stage_history", []), "completed"]
        }

    # =========================================================================
    # Public API
    # =========================================================================

    async def run(
        self,
        initial_state: Dict[str, Any],
        thread_id: Optional[str] = None
    ) -> HarvesterV2State:
        """
        Execute the intelligence workflow.

        Args:
            initial_state: Initial state with customer_id, tenant_id, context
            thread_id: Thread ID for checkpointing (optional)

        Returns:
            Final state with all intelligence results
        """
        if not LANGGRAPH_AVAILABLE or self.graph is None:
            logger.warning("LangGraph not available, running simplified flow")
            return await self._run_simplified(initial_state)

        # Prepare initial state
        workflow_id = initial_state.get("workflow_id", str(uuid.uuid4()))
        customer_id = initial_state["customer_id"]
        tenant_id = initial_state.get("tenant_id", self.tenant_id)
        session_id = initial_state.get("session_id", "")

        # Create Langfuse trace if enabled (before state construction for immutability)
        langfuse_trace_id = None
        if self.tracer:
            try:
                trace = self.tracer.trace(
                    name="harvester_v2_intelligence",
                    session_id=session_id,
                    metadata={
                        "tenant_id": tenant_id,
                        "customer_id": customer_id,
                        "workflow_id": workflow_id
                    }
                )
                langfuse_trace_id = trace.id
            except Exception as e:
                logger.debug("Failed to create Langfuse trace", error=str(e))

        # Build immutable state
        state: HarvesterV2State = {
            "workflow_id": workflow_id,
            "customer_id": customer_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "_langfuse_trace_id": langfuse_trace_id,
            "input_context": initial_state.get("context", {}),
            "memory_snapshot": {},
            "scoring_result": None,
            "causality_result": None,
            "reasoning_result": None,
            "nba_result": None,
            "retention_result": None,
            "final_intelligence": None,
            "current_stage": "pending",
            "stage_history": [],
            "errors": [],
            "critique_score": 0.0,
            "iteration_count": 0
        }

        # Get checkpointer if available
        checkpointer = None
        if self.use_checkpointing and ADAPTER_AVAILABLE:
            try:
                adapter = await get_async_langgraph_adapter(use_postgres=True)
                checkpointer = await adapter.get_async_checkpointer()
            except Exception as e:
                logger.warning("Checkpointing not available", error=str(e))

        # Compile graph with checkpointer
        compiled = self.graph.compile(checkpointer=checkpointer)

        # Execute
        config = {
            "configurable": {
                "thread_id": thread_id or state.get("session_id") or workflow_id
            }
        }

        try:
            result = await compiled.ainvoke(state, config=config)

            # End Langfuse trace if available
            if self.tracer and state.get("_langfuse_trace_id"):
                try:
                    self.tracer.flush()
                except Exception as e:
                    logger.debug("Langfuse flush failed", error=str(e))

            return result

        except Exception as e:
            logger.error(
                "Workflow execution failed",
                workflow_id=workflow_id,
                error=str(e)
            )
            raise

    async def _run_simplified(
        self,
        initial_state: Dict[str, Any]
    ) -> HarvesterV2State:
        """
        Simplified execution without LangGraph.

        Used as fallback when LangGraph is not available.
        """
        workflow_id = str(uuid.uuid4())

        state: HarvesterV2State = {
            "workflow_id": workflow_id,
            "customer_id": initial_state["customer_id"],
            "tenant_id": initial_state.get("tenant_id", self.tenant_id),
            "session_id": initial_state.get("session_id", ""),
            "_langfuse_trace_id": None,
            "input_context": initial_state.get("context", {}),
            "memory_snapshot": {},
            "scoring_result": None,
            "causality_result": None,
            "reasoning_result": None,
            "nba_result": None,
            "retention_result": None,
            "final_intelligence": None,
            "current_stage": "pending",
            "stage_history": [],
            "errors": [],
            "critique_score": 0.0,
            "iteration_count": 0
        }

        # Run each stage sequentially
        state.update(await self._initialize_node(state))
        state.update(await self._scoring_node(state))
        state.update(await self._analysis_node(state))
        state.update(await self._actions_node(state))
        state.update(await self._aggregate_node(state))
        state.update(await self._critique_node(state))
        state.update(await self._complete_node(state))

        return state
