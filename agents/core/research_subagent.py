"""
Research Sub-Agent for Harvester V2
====================================

Web search and synthesis sub-agent that enriches customer intelligence
with external market research and competitor analysis.

Features:
- Generates context-aware search queries
- Parallel search across Tavily, DuckDuckGo, Jina
- Content extraction via Firecrawl
- LLM-powered synthesis of findings
- **REFLECTION LOOP**: Iterative deep research (EnterpriseGPT pattern)

The reflection loop enables Claude Code-style deep research:
1. Execute initial search queries
2. Reflect on findings - are they sufficient?
3. If not, generate follow-up queries based on knowledge gaps
4. Search again and merge results
5. Final synthesis

This brings EnterpriseGPT-style research capabilities into
Harvester V2's intelligence pipeline.

Usage:
    subagent = ResearchSubAgent()
    result = await subagent.run_isolated(
        context,
        "Research market context for customer"
    )
"""

from typing import Any, Dict, List, Optional
import json
import os

from harvester_v2.agents.base_subagent import BaseSubAgent, SubAgentContext
from harvester_v2.agents.parallel_executor import SubAgentRegistry
from shared.ai.model_router import TaskType
from shared.logging import get_platform_logger

logger = get_platform_logger(__name__)

# Lazy import to avoid circular dependencies
_search_orchestrator = None


def get_search_orchestrator():
    """Get or create the search orchestrator instance."""
    global _search_orchestrator
    if _search_orchestrator is None:
        try:
            from harvester_v2.search.orchestrator import WebSearchOrchestrator
            _search_orchestrator = WebSearchOrchestrator({
                "search_engines": ["tavily", "duckduckgo"],
                "max_results_per_engine": 5,
                "extractors": [],  # Firecrawl optional
                "max_urls_to_extract": 0  # Skip extraction for speed
            })
            logger.info(
                "ResearchSubAgent search orchestrator initialized",
                engines=_search_orchestrator.get_available_engines()
            )
        except Exception as e:
            logger.warning(
                "Search orchestrator initialization failed",
                error=str(e)
            )
            _search_orchestrator = None
    return _search_orchestrator


@SubAgentRegistry.register
class ResearchSubAgent(BaseSubAgent):
    """
    Web search and synthesis sub-agent.

    Generates search queries based on customer context, executes
    parallel web search, and synthesizes findings into actionable
    research insights.

    Outputs:
    1. search_queries: Queries used for web search
    2. source_urls: URLs discovered during search
    3. search_summaries: Per-query result summaries
    4. research_synthesis: LLM-synthesized findings
    5. research_confidence: Confidence in findings
    """

    agent_name = "research_subagent"

    # Standard critique threshold
    critique_threshold = 0.85

    def __init__(self, options: Dict[str, Any] = None):
        super().__init__(agent_name="research_subagent", options=options or {})
        self.max_queries = 5
        self.max_reflection_iterations = 2  # Max follow-up search rounds
        self.reflection_threshold = 0.7  # Below this triggers follow-up
        self.search_orchestrator = None

    def _ensure_orchestrator(self) -> bool:
        """Ensure search orchestrator is available."""
        if self.search_orchestrator is None:
            self.search_orchestrator = get_search_orchestrator()
        return self.search_orchestrator is not None

    async def _plan(self, context: SubAgentContext, task: str) -> Dict[str, Any]:
        """Plan research by generating targeted search queries."""
        input_data = context.input_data or {}

        # Extract relevant context for query generation
        industry = input_data.get("industry", "")
        product = input_data.get("product", "")
        competitor_mentions = input_data.get("competitor_mentions", [])
        churn_signals = input_data.get("churn_signals", [])
        recent_issues = []
        for interaction in input_data.get("recent_interactions", []):
            if interaction.get("sentiment") == "negative":
                recent_issues.append(interaction.get("issue", ""))

        prompt = f"""You are a research strategist. Generate 3-5 targeted search queries
to gather external market intelligence for this customer scenario.

CUSTOMER CONTEXT:
- Industry: {industry}
- Product: {product}
- Competitor Mentions: {competitor_mentions}
- Churn Signals: {churn_signals}
- Recent Issues: {recent_issues}

Generate queries that will find:
1. Industry best practices for customer retention
2. Competitor analysis and comparisons
3. Solutions for the specific issues faced
4. Market trends relevant to this customer

Respond with ONLY a JSON object (no markdown):
{{
    "search_queries": [
        "query 1",
        "query 2",
        "query 3"
    ],
    "research_focus": "Brief description of what we're looking for",
    "expected_insights": ["insight type 1", "insight type 2"]
}}"""

        try:
            response = await self.model_router.complete(
                prompt=prompt,
                task_type=TaskType.REASONING,
                agent_name=f"{self.agent_name}_planner",
                tenant_id=context.tenant_id
            )

            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            plan = json.loads(response_text)
            plan["search_queries"] = plan.get("search_queries", [])[:self.max_queries]
            return plan

        except json.JSONDecodeError as e:
            logger.warning(
                "Plan JSON parsing failed, using fallback queries",
                error=str(e),
                customer_id=context.customer_id
            )
            # Fallback queries based on context
            queries = []
            if industry:
                queries.append(f"{industry} customer retention best practices 2025")
            if product:
                queries.append(f"{product} service issues resolution")
            if competitor_mentions:
                queries.append(f"{competitor_mentions[0]} vs alternatives comparison")
            if not queries:
                queries = ["B2B customer retention strategies", "churn prevention tactics"]

            return {
                "search_queries": queries,
                "research_focus": "General market research",
                "expected_insights": ["retention strategies", "competitor insights"]
            }

        except Exception as e:
            logger.error("Research planning failed", error=str(e))
            return {
                "search_queries": ["customer retention strategies"],
                "research_focus": "Fallback research",
                "expected_insights": [],
                "error": str(e)
            }

    async def _execute_plan(self, context: SubAgentContext, plan: Dict) -> Dict[str, Any]:
        """Execute web search with reflection loop for deep research."""
        search_queries = plan.get("search_queries", [])

        if not search_queries:
            return self._get_fallback_result(context, "No search queries generated")

        # Check if search orchestrator is available
        if not self._ensure_orchestrator():
            return self._get_fallback_result(
                context,
                "Search orchestrator not available (API keys may be missing)"
            )

        all_urls: List[str] = []
        search_summaries: List[Dict[str, Any]] = []
        reflection_iterations = 0
        all_queries_executed = []

        # Initial search
        urls, summaries = await self._execute_search_queries(search_queries)
        all_urls.extend(urls)
        search_summaries.extend(summaries)
        all_queries_executed.extend(search_queries)

        # REFLECTION LOOP: Check if research is sufficient
        while reflection_iterations < self.max_reflection_iterations:
            # Reflect on current findings
            reflection = await self._reflect_on_findings(
                context,
                all_queries_executed,
                search_summaries,
                all_urls
            )

            if reflection.get("is_sufficient", True):
                logger.info(
                    "Research sufficient after reflection",
                    iterations=reflection_iterations,
                    urls_found=len(all_urls),
                    customer_id=context.customer_id
                )
                break

            # Generate follow-up queries based on knowledge gaps
            follow_up_queries = reflection.get("follow_up_queries", [])
            if not follow_up_queries:
                logger.info(
                    "No follow-up queries generated, stopping reflection",
                    iterations=reflection_iterations
                )
                break

            logger.info(
                "Reflection identified knowledge gaps, executing follow-up queries",
                gaps=reflection.get("knowledge_gaps", []),
                follow_up_count=len(follow_up_queries),
                iteration=reflection_iterations + 1
            )

            # Execute follow-up search
            new_urls, new_summaries = await self._execute_search_queries(follow_up_queries)
            all_urls.extend(new_urls)
            search_summaries.extend(new_summaries)
            all_queries_executed.extend(follow_up_queries)

            reflection_iterations += 1

        # Deduplicate URLs
        unique_urls = list(set(all_urls))

        # Final synthesis
        synthesis = await self._synthesize_findings(
            context,
            all_queries_executed,
            search_summaries,
            unique_urls
        )

        return {
            "search_queries": all_queries_executed,
            "source_urls": unique_urls[:30],  # Increased limit for deep research
            "search_summaries": search_summaries,
            "research_synthesis": synthesis.get("synthesis", ""),
            "key_findings": synthesis.get("key_findings", []),
            "research_confidence": synthesis.get("confidence", 0.7),
            "reflection_iterations": reflection_iterations,
            "customer_id": context.customer_id,
            "tenant_id": context.tenant_id
        }

    async def _execute_search_queries(
        self,
        queries: List[str]
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Execute a batch of search queries and return results."""
        all_urls: List[str] = []
        search_summaries: List[Dict[str, Any]] = []

        for query in queries:
            try:
                results_text, urls = await self.search_orchestrator.orchestrated_search(
                    query,
                    extract_content=False  # Skip extraction for speed
                )
                all_urls.extend(urls)
                search_summaries.append({
                    "query": query,
                    "results_count": len(urls),
                    "snippet": results_text[:500] if results_text else ""
                })
                logger.debug(
                    "Search completed",
                    query=query[:50],
                    urls_found=len(urls)
                )
            except Exception as e:
                logger.warning(
                    "Search query failed",
                    query=query,
                    error=str(e)
                )
                search_summaries.append({
                    "query": query,
                    "results_count": 0,
                    "error": str(e)
                })

        return all_urls, search_summaries

    async def _reflect_on_findings(
        self,
        context: SubAgentContext,
        queries_executed: List[str],
        summaries: List[Dict[str, Any]],
        urls: List[str]
    ) -> Dict[str, Any]:
        """
        Reflect on current research findings to identify gaps.

        This is the core of the EnterpriseGPT-style deep research pattern.
        We ask the LLM to evaluate if the current findings are sufficient
        for the customer intelligence task, and if not, what follow-up
        queries would fill the gaps.
        """
        input_data = context.input_data or {}

        prompt = f"""You are a research quality evaluator. Assess if the current research
findings are sufficient for customer intelligence analysis.

CUSTOMER CONTEXT:
- Industry: {input_data.get('industry', 'Unknown')}
- Product: {input_data.get('product', 'Unknown')}
- Key Issues: {input_data.get('churn_signals', [])}
- Competitors: {input_data.get('competitor_mentions', [])}

QUERIES EXECUTED:
{json.dumps(queries_executed, indent=2)}

SEARCH RESULTS SUMMARY:
{json.dumps(summaries, indent=2, default=str)[:3000]}

SOURCES FOUND: {len(urls)} URLs

EVALUATION CRITERIA:
1. Do we have industry-specific retention strategies?
2. Do we understand the competitive landscape?
3. Do we have data on the specific issues mentioned?
4. Are there significant gaps in our research?

Respond with ONLY a JSON object (no markdown):
{{
    "is_sufficient": true|false,
    "sufficiency_score": 0.X,
    "knowledge_gaps": [
        "Gap 1: What information is missing",
        "Gap 2: What information is missing"
    ],
    "follow_up_queries": [
        "More specific query to fill gap 1",
        "More specific query to fill gap 2"
    ],
    "reasoning": "Why the research is or isn't sufficient"
}}"""

        try:
            response = await self.model_router.complete(
                prompt=prompt,
                task_type=TaskType.REASONING,
                agent_name=f"{self.agent_name}_reflector",
                tenant_id=context.tenant_id
            )

            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            reflection = json.loads(response_text)

            # Apply threshold logic
            score = reflection.get("sufficiency_score", 0.8)
            if score < self.reflection_threshold:
                reflection["is_sufficient"] = False

            # Limit follow-up queries
            reflection["follow_up_queries"] = reflection.get("follow_up_queries", [])[:3]

            return reflection

        except json.JSONDecodeError as e:
            logger.warning(
                "Reflection JSON parsing failed",
                error=str(e)
            )
            return {"is_sufficient": True, "reasoning": f"Reflection failed: {e}"}

        except Exception as e:
            logger.warning(
                "Reflection failed, assuming sufficient",
                error=str(e)
            )
            return {"is_sufficient": True, "reasoning": f"Reflection error: {e}"}

    async def _synthesize_findings(
        self,
        context: SubAgentContext,
        queries: List[str],
        summaries: List[Dict],
        urls: List[str]
    ) -> Dict[str, Any]:
        """Synthesize search results into actionable insights."""
        input_data = context.input_data or {}

        prompt = f"""You are a research analyst synthesizing web search findings
for customer intelligence.

CUSTOMER CONTEXT:
{json.dumps(input_data, indent=2, default=str)[:2000]}

SEARCH QUERIES EXECUTED:
{json.dumps(queries, indent=2)}

SEARCH RESULTS SUMMARIES:
{json.dumps(summaries, indent=2, default=str)[:3000]}

SOURCES FOUND ({len(urls)} URLs):
{json.dumps(urls[:10], indent=2)}

Synthesize these findings into actionable research insights.
Focus on:
1. Industry best practices relevant to this customer
2. Competitive landscape and positioning
3. Specific recommendations backed by research
4. Data points that strengthen our analysis

Respond with ONLY a JSON object (no markdown):
{{
    "synthesis": "A 2-3 paragraph synthesis of key findings that directly apply to this customer's situation...",
    "key_findings": [
        "Key finding 1 with specific data or insight",
        "Key finding 2 with specific data or insight",
        "Key finding 3 with specific data or insight"
    ],
    "confidence": 0.X,
    "research_quality": "high|medium|low"
}}"""

        try:
            response = await self.model_router.complete(
                prompt=prompt,
                task_type=TaskType.REASONING,
                agent_name=f"{self.agent_name}_synthesizer",
                tenant_id=context.tenant_id
            )

            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            return json.loads(response_text)

        except json.JSONDecodeError as e:
            logger.warning(
                "Synthesis JSON parsing failed",
                error=str(e)
            )
            return {
                "synthesis": f"Web research completed with {len(urls)} sources found across {len(queries)} queries. Manual review recommended.",
                "key_findings": [],
                "confidence": 0.5,
                "research_quality": "low"
            }

        except Exception as e:
            logger.error("Research synthesis failed", error=str(e))
            return {
                "synthesis": f"Research synthesis unavailable: {e}",
                "key_findings": [],
                "confidence": 0.3,
                "error": str(e)
            }

    def _get_fallback_result(
        self,
        context: SubAgentContext,
        error: str
    ) -> Dict[str, Any]:
        """Return fallback result when research fails."""
        return {
            "search_queries": [],
            "source_urls": [],
            "search_summaries": [],
            "research_synthesis": f"Research unavailable: {error}",
            "key_findings": [],
            "research_confidence": 0.0,
            "customer_id": context.customer_id,
            "tenant_id": context.tenant_id,
            "error": error
        }
