"""
Base Sub-Agent Class for Harvester V2
======================================

Implements the DeepAgents pattern with:
- Isolated execution contexts (fresh for each run)
- Plan → Execute → Critique → Output lifecycle
- Automatic tracing via Langfuse
- Model router integration (tenant-aware)
- Automatic iteration on low critique scores

This differs from BaseAgent in that:
1. Each run gets a fresh, isolated context (not shared UnifiedState)
2. Sub-agents are called by the orchestrator, not directly by users
3. Results are returned as structured SubAgentResult objects
4. Contexts are never mutated - new contexts are created for each run

Usage:
    class MyScoringSubAgent(BaseSubAgent):
        async def _plan(self, context, task):
            return {"steps": ["calculate_fraud", "calculate_churn"]}

        async def _execute_plan(self, context, plan):
            return {"fraud_score": 0.2, "churn_risk": 0.7}

    # Execute with isolated context
    subagent = MyScoringSubAgent()
    result = await subagent.run_isolated(context, "Score customer risk")
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import time
import json
import structlog

from shared.ai.model_router import get_model_router, TaskType
from shared.logging import get_platform_logger
from shared.errors import AgentExecutionError

# Try to import Langfuse tracer (optional)
try:
    from shared.observability import get_langfuse_tracer
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    get_langfuse_tracer = None

logger = get_platform_logger(__name__)


class SubAgentStage(Enum):
    """Execution stages for sub-agent lifecycle."""
    PLAN = "plan"
    EXECUTE = "execute"
    CRITIQUE = "critique"
    REFINE = "refine"
    OUTPUT = "output"


@dataclass
class SubAgentContext:
    """
    Isolated execution context for a sub-agent.

    Each sub-agent run gets a fresh context that is never mutated.
    This ensures complete isolation between runs.

    Attributes:
        context_id: Unique identifier for this context
        tenant_id: Tenant identifier for multi-tenant isolation
        customer_id: Customer being analyzed (optional)
        input_data: Input data for this run
        memory_snapshot: Relevant memories loaded at start
        session_id: Session this run is part of (optional)
        orchestrator_run_id: ID of the orchestrator run (optional)
        created_at: When this context was created
        squeezed_memory_snapshot: LLM-compressed summary of memories
        token_budget: Maximum tokens allowed for this context
        squeeze_metadata: Compression statistics and metadata
        adaptive_window: Per-agent token configuration
    """
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    customer_id: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    orchestrator_run_id: Optional[str] = None
    langfuse_trace_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    # Context squeezing fields
    squeezed_memory_snapshot: Optional[str] = None
    token_budget: int = 40000  # Default 40K tokens
    squeeze_metadata: Dict[str, Any] = field(default_factory=dict)
    adaptive_window: Optional[Dict[str, Any]] = None

    def to_prompt_context(self) -> str:
        """
        Convert context to a prompt-friendly string format.

        Uses squeezed memory if available, otherwise falls back to raw snapshot.

        Returns:
            Formatted string for inclusion in LLM prompts
        """
        lines = [
            f"Context ID: {self.context_id}",
            f"Tenant: {self.tenant_id}",
            f"Customer: {self.customer_id or 'N/A'}",
            f"Created: {self.created_at.isoformat()}",
            "",
            "INPUT DATA:",
            json.dumps(self.input_data, indent=2, default=str),
            "",
            "RELEVANT MEMORY:",
        ]

        # Prefer squeezed context for token efficiency
        if self.squeezed_memory_snapshot:
            lines.append(self.squeezed_memory_snapshot)
            if self.squeeze_metadata:
                ratio = self.squeeze_metadata.get("compression_ratio", 0)
                lines.append(f"\n(Compressed: {ratio:.0%} reduction)")
        elif self.memory_snapshot:
            lines.append(json.dumps(self.memory_snapshot, indent=2, default=str))
        else:
            lines.append("(none loaded)")

        return "\n".join(lines)

    def copy(self) -> "SubAgentContext":
        """
        Create a deep copy of this context.

        Returns:
            New SubAgentContext with copied data
        """
        return SubAgentContext(
            context_id=str(uuid.uuid4()),  # New ID for the copy
            tenant_id=self.tenant_id,
            customer_id=self.customer_id,
            input_data=self.input_data.copy() if self.input_data else {},
            memory_snapshot=self.memory_snapshot.copy() if self.memory_snapshot else {},
            session_id=self.session_id,
            orchestrator_run_id=self.orchestrator_run_id,
            langfuse_trace_id=self.langfuse_trace_id,
            created_at=datetime.utcnow(),
            # Copy context squeezing fields
            squeezed_memory_snapshot=self.squeezed_memory_snapshot,
            token_budget=self.token_budget,
            squeeze_metadata=self.squeeze_metadata.copy() if self.squeeze_metadata else {},
            adaptive_window=self.adaptive_window.copy() if self.adaptive_window else None
        )


@dataclass
class SubAgentResult:
    """
    Standardized result from a sub-agent execution.

    All sub-agents return this structured result to enable
    consistent aggregation by the orchestrator.

    Attributes:
        subagent_name: Name of the sub-agent that produced this result
        success: Whether execution succeeded
        data: Output data from the sub-agent
        reasoning_trace: Explanation of the reasoning process
        critique_score: Quality score from critique stage (0-1)
        execution_time_ms: Total execution time in milliseconds
        model_used: Model that was used for this execution
        iterations: Number of iterations (critique + refine cycles)
        error: Error message if execution failed
        metadata: Additional metadata (token counts, etc.)
        evaluation: Evaluation metrics (grounding_score, hallucination_flags, trajectory_score)
    """
    subagent_name: str
    success: bool
    data: Dict[str, Any]
    reasoning_trace: str
    critique_score: float
    execution_time_ms: int
    model_used: str
    iterations: int = 1
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Evaluation metrics
    evaluation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "subagent_name": self.subagent_name,
            "success": self.success,
            "data": self.data,
            "reasoning_trace": self.reasoning_trace,
            "critique_score": self.critique_score,
            "execution_time_ms": self.execution_time_ms,
            "model_used": self.model_used,
            "iterations": self.iterations,
            "error": self.error,
            "metadata": self.metadata,
            "evaluation": self.evaluation
        }


class BaseSubAgent(ABC):
    """
    Base class for Harvester V2 sub-agents.

    Implements the DeepAgents pattern:
    1. PLAN: Generate execution plan from context and task
    2. EXECUTE: Run the plan to produce output
    3. CRITIQUE: Evaluate output quality
    4. REFINE: Improve output if critique score is low (iterate)
    5. OUTPUT: Return structured SubAgentResult

    Key features:
    - Isolated context (fresh for each execution)
    - Automatic critique and iteration
    - Langfuse tracing integration
    - Model router for tenant-aware model selection

    Subclasses must implement:
    - _plan(context, task) -> Dict
    - _execute_plan(context, plan) -> Dict

    Optional overrides:
    - _critique(context, task, result) -> Tuple[score, reasoning]
    - _refine(context, result, critique) -> Dict
    """

    # Quality threshold for accepting results (0.85 = 85%)
    critique_threshold: float = 0.85

    # Maximum iterations (plan-execute-critique cycles)
    max_iterations: int = 3

    # Default task type for model routing
    default_task_type: TaskType = TaskType.REASONING

    def __init__(
        self,
        agent_name: str,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize sub-agent.

        Args:
            agent_name: Unique name for this sub-agent
            options: Optional configuration overrides
        """
        self.agent_name = agent_name
        self.options = options or {}

        # Initialize model router
        self.model_router = get_model_router()

        # Initialize Langfuse tracer if available
        self.tracer = None
        if LANGFUSE_AVAILABLE and get_langfuse_tracer:
            try:
                self.tracer = get_langfuse_tracer()
            except Exception as e:
                logger.debug("Langfuse tracer not configured", error=str(e))

        # Override thresholds from options
        if "critique_threshold" in self.options:
            self.critique_threshold = self.options["critique_threshold"]
        if "max_iterations" in self.options:
            self.max_iterations = self.options["max_iterations"]

        logger.info(
            "Sub-agent initialized",
            subagent=agent_name,
            critique_threshold=self.critique_threshold,
            max_iterations=self.max_iterations
        )

    async def run_isolated(
        self,
        context: SubAgentContext,
        task_description: str
    ) -> SubAgentResult:
        """
        Execute sub-agent with isolated context.

        This is the main entry point for sub-agent execution.
        Each call gets a fresh context - no pollution from previous runs.

        The lifecycle is:
        1. PLAN: Generate execution plan
        2. EXECUTE: Run the plan
        3. CRITIQUE: Evaluate quality
        4. ITERATE: Refine if below threshold (up to max_iterations)
        5. OUTPUT: Return structured result

        Args:
            context: Isolated execution context
            task_description: Description of the task to perform

        Returns:
            SubAgentResult with output data and metadata
        """
        start_time = time.time()
        iterations = 0

        # Create trace span if Langfuse is available
        span = None
        if self.tracer and context.langfuse_trace_id:
            try:
                span = self.tracer.span(
                    trace_id=context.langfuse_trace_id,
                    name=f"subagent_{self.agent_name}",
                    input_data={
                        "context_id": context.context_id,
                        "task": task_description,
                        "customer_id": context.customer_id
                    }
                )
            except Exception as e:
                logger.debug("Failed to create Langfuse span", error=str(e))

        try:
            # Log start
            logger.info(
                "Sub-agent execution started",
                subagent=self.agent_name,
                context_id=context.context_id,
                customer_id=context.customer_id,
                tenant_id=context.tenant_id
            )

            # PLAN: Generate execution plan
            plan = await self._plan_with_tracing(context, task_description, span)

            # EXECUTE: Run the plan
            result_data = await self._execute_with_tracing(context, plan, span)

            # CRITIQUE: Evaluate quality
            critique_score, reasoning = await self._critique_with_tracing(
                context, task_description, result_data, span
            )

            iterations = 1

            # ITERATE: Refine if below threshold
            while critique_score < self.critique_threshold and iterations < self.max_iterations:
                logger.info(
                    "Critique below threshold, refining",
                    subagent=self.agent_name,
                    score=critique_score,
                    threshold=self.critique_threshold,
                    iteration=iterations + 1
                )

                # REFINE: Improve result based on critique
                result_data = await self._refine_with_tracing(
                    context, result_data, reasoning, span
                )

                # Re-critique
                critique_score, reasoning = await self._critique_with_tracing(
                    context, task_description, result_data, span
                )

                iterations += 1

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Build result
            result = SubAgentResult(
                subagent_name=self.agent_name,
                success=True,
                data=result_data,
                reasoning_trace=reasoning,
                critique_score=critique_score,
                execution_time_ms=execution_time_ms,
                model_used=self._get_model_name(context.tenant_id),
                iterations=iterations,
                metadata={
                    "context_id": context.context_id,
                    "plan": plan
                }
            )

            # Log completion
            logger.info(
                "Sub-agent execution completed",
                subagent=self.agent_name,
                context_id=context.context_id,
                success=True,
                critique_score=critique_score,
                iterations=iterations,
                execution_time_ms=execution_time_ms
            )

            # End trace span
            if span:
                try:
                    span.end(
                        output=result.to_dict(),
                        status="success"
                    )
                except Exception as e:
                    logger.debug("Langfuse span.end failed", error=str(e))

            return result

        except Exception as e:
            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Log error
            logger.error(
                "Sub-agent execution failed",
                subagent=self.agent_name,
                context_id=context.context_id,
                error=str(e)
            )

            # End trace span with error
            if span:
                try:
                    span.end(
                        output={"error": str(e)},
                        status="error"
                    )
                except Exception as trace_err:
                    logger.debug("Langfuse span.end failed", error=str(trace_err))

            return SubAgentResult(
                subagent_name=self.agent_name,
                success=False,
                data={},
                reasoning_trace="",
                critique_score=0.0,
                execution_time_ms=execution_time_ms,
                model_used="",
                iterations=iterations,
                error=str(e)
            )

    # =========================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    async def _plan(
        self,
        context: SubAgentContext,
        task: str
    ) -> Dict[str, Any]:
        """
        Generate execution plan from context and task.

        This method should analyze the input and decide:
        - What steps to execute
        - What data to use
        - What the expected output format is

        Args:
            context: Isolated execution context
            task: Task description

        Returns:
            Plan dictionary with execution steps
        """
        pass

    @abstractmethod
    async def _execute_plan(
        self,
        context: SubAgentContext,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the plan to produce output.

        This method contains the core business logic of the sub-agent.
        It should execute the plan steps and return structured output.

        Args:
            context: Isolated execution context
            plan: Execution plan from _plan()

        Returns:
            Output dictionary with results
        """
        pass

    # =========================================================================
    # Optional Methods (Can be overridden for custom behavior)
    # =========================================================================

    async def _critique(
        self,
        context: SubAgentContext,
        task: str,
        result: Dict[str, Any]
    ) -> Tuple[float, str]:
        """
        Critique the result quality.

        Default implementation uses LLM to evaluate:
        1. Completeness: Does it address all aspects?
        2. Accuracy: Is the reasoning sound?
        3. Actionability: Can this be used for decisions?

        Override for custom critique logic.

        Args:
            context: Isolated execution context
            task: Original task description
            result: Execution result to critique

        Returns:
            Tuple of (score 0-1, reasoning string)
        """
        prompt = f"""You are a quality critic evaluating sub-agent output.

TASK: {task}

CONTEXT:
{context.to_prompt_context()}

RESULT:
{json.dumps(result, indent=2, default=str)}

Evaluate the result on these criteria:
1. Completeness (0-1): Does it address all aspects of the task?
2. Accuracy (0-1): Is the reasoning sound and data correct?
3. Actionability (0-1): Can this be used to make decisions?

Respond with ONLY a JSON object (no markdown, no explanation):
{{
    "completeness": 0.X,
    "accuracy": 0.X,
    "actionability": 0.X,
    "overall_score": 0.X,
    "reasoning": "Brief explanation of the score"
}}"""

        try:
            response = await self.model_router.complete(
                prompt=prompt,
                task_type=TaskType.REASONING,
                agent_name=f"{self.agent_name}_critic",
                tenant_id=context.tenant_id
            )

            # Parse JSON response
            # Handle markdown code blocks if present
            response_text = response.strip()
            if response_text.startswith("```"):
                # Remove markdown code block
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text

            critique = json.loads(response_text)
            return critique.get("overall_score", 0.5), critique.get("reasoning", "")

        except json.JSONDecodeError:
            # If can't parse JSON, return moderate score
            logger.warning(
                "Failed to parse critique JSON, using default",
                subagent=self.agent_name,
                response=response[:200] if response else ""
            )
            return 0.7, response if response else "Could not parse critique"

        except Exception as e:
            logger.warning(
                "Critique failed, using default score",
                subagent=self.agent_name,
                error=str(e)
            )
            return 0.7, f"Critique failed: {str(e)}"

    async def _refine(
        self,
        context: SubAgentContext,
        result: Dict[str, Any],
        critique: str
    ) -> Dict[str, Any]:
        """
        Refine result based on critique feedback.

        Default implementation asks LLM to improve the result.
        Override for custom refinement logic.

        Args:
            context: Isolated execution context
            result: Current result to improve
            critique: Critique reasoning

        Returns:
            Improved result dictionary
        """
        prompt = f"""You are improving sub-agent output based on critique feedback.

CURRENT RESULT:
{json.dumps(result, indent=2, default=str)}

CRITIQUE:
{critique}

CONTEXT:
{context.to_prompt_context()}

Improve the result to address the critique. Keep the same structure but enhance:
- Completeness: Fill in any gaps
- Accuracy: Fix any reasoning errors
- Actionability: Make recommendations clearer

Respond with ONLY the improved JSON result (no markdown, no explanation):"""

        try:
            response = await self.model_router.complete(
                prompt=prompt,
                task_type=TaskType.REASONING,
                agent_name=f"{self.agent_name}_refiner",
                tenant_id=context.tenant_id
            )

            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text

            refined = json.loads(response_text)
            return refined

        except Exception as e:
            logger.warning(
                "Refinement failed, keeping original",
                subagent=self.agent_name,
                error=str(e)
            )
            return result

    # =========================================================================
    # Internal Methods (Tracing wrappers)
    # =========================================================================

    async def _plan_with_tracing(
        self,
        context: SubAgentContext,
        task: str,
        parent_span: Any = None
    ) -> Dict[str, Any]:
        """Plan with Langfuse tracing."""
        span = None
        if self.tracer and parent_span:
            try:
                span = parent_span.span(
                    name=f"{self.agent_name}_plan",
                    input_data={"task": task}
                )
            except Exception as e:
                logger.debug("Langfuse span creation failed", stage="plan", error=str(e))

        start = time.time()
        result = await self._plan(context, task)
        duration_ms = int((time.time() - start) * 1000)

        if span:
            try:
                span.end(output=result, metadata={"duration_ms": duration_ms})
            except Exception as e:
                logger.debug("Langfuse span.end failed", stage="plan", error=str(e))

        logger.debug(
            "Plan stage completed",
            subagent=self.agent_name,
            duration_ms=duration_ms
        )

        return result

    async def _execute_with_tracing(
        self,
        context: SubAgentContext,
        plan: Dict[str, Any],
        parent_span: Any = None
    ) -> Dict[str, Any]:
        """Execute with Langfuse tracing."""
        span = None
        if self.tracer and parent_span:
            try:
                span = parent_span.span(
                    name=f"{self.agent_name}_execute",
                    input_data={"plan": plan}
                )
            except Exception as e:
                logger.debug("Langfuse span creation failed", stage="execute", error=str(e))

        start = time.time()
        result = await self._execute_plan(context, plan)
        duration_ms = int((time.time() - start) * 1000)

        if span:
            try:
                span.end(output=result, metadata={"duration_ms": duration_ms})
            except Exception as e:
                logger.debug("Langfuse span.end failed", stage="execute", error=str(e))

        logger.debug(
            "Execute stage completed",
            subagent=self.agent_name,
            duration_ms=duration_ms
        )

        return result

    async def _critique_with_tracing(
        self,
        context: SubAgentContext,
        task: str,
        result: Dict[str, Any],
        parent_span: Any = None
    ) -> Tuple[float, str]:
        """Critique with Langfuse tracing."""
        span = None
        if self.tracer and parent_span:
            try:
                span = parent_span.span(
                    name=f"{self.agent_name}_critique",
                    input_data={"result_keys": list(result.keys())}
                )
            except Exception as e:
                logger.debug("Langfuse span creation failed", stage="critique", error=str(e))

        start = time.time()
        score, reasoning = await self._critique(context, task, result)
        duration_ms = int((time.time() - start) * 1000)

        if span:
            try:
                span.end(
                    output={"score": score, "reasoning": reasoning[:200]},
                    metadata={"duration_ms": duration_ms}
                )
            except Exception as e:
                logger.debug("Langfuse span.end failed", stage="critique", error=str(e))

        logger.debug(
            "Critique stage completed",
            subagent=self.agent_name,
            score=score,
            duration_ms=duration_ms
        )

        return score, reasoning

    async def _refine_with_tracing(
        self,
        context: SubAgentContext,
        result: Dict[str, Any],
        critique: str,
        parent_span: Any = None
    ) -> Dict[str, Any]:
        """Refine with Langfuse tracing."""
        span = None
        if self.tracer and parent_span:
            try:
                span = parent_span.span(
                    name=f"{self.agent_name}_refine",
                    input_data={"critique": critique[:200]}
                )
            except Exception as e:
                logger.debug("Langfuse span creation failed", stage="refine", error=str(e))

        start = time.time()
        refined = await self._refine(context, result, critique)
        duration_ms = int((time.time() - start) * 1000)

        if span:
            try:
                span.end(
                    output={"refined_keys": list(refined.keys())},
                    metadata={"duration_ms": duration_ms}
                )
            except Exception as e:
                logger.debug("Langfuse span.end failed", stage="refine", error=str(e))

        logger.debug(
            "Refine stage completed",
            subagent=self.agent_name,
            duration_ms=duration_ms
        )

        return refined

    def _get_model_name(self, tenant_id: str) -> str:
        """
        Get model name for tracing.

        Override to customize model selection per tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Model name string
        """
        # Default: use the model from options or fallback
        return self.options.get("model", "claude-sonnet")
