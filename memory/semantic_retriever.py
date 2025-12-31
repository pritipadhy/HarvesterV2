"""
Semantic Retriever for Harvester V2
====================================

RAG (Retrieval-Augmented Generation) for domain knowledge.
Retrieves relevant facts, policies, procedures, and playbooks
to augment sub-agent reasoning.

Knowledge categories:
- product: Product information, features, pricing
- policy: Business policies, rules, constraints
- procedure: Standard operating procedures
- faq: Frequently asked questions and answers
- playbook: Success playbooks for specific scenarios

Usage:
    retriever = SemanticRetriever(tenant_id="gigaclear")

    # Get relevant knowledge for a customer context
    knowledge = await retriever.get_relevant_knowledge(
        customer_id="CUST-123",
        query="What retention offers are available?"
    )

    # Search across all knowledge
    results = await retriever.search("broadband upgrade policy")
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import structlog

from shared.logging import get_platform_logger

# Try to import Weaviate
try:
    from shared.database.weaviate_client import create_weaviate_client
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

logger = get_platform_logger(__name__)


@dataclass
class KnowledgeItem:
    """
    A single knowledge item.

    Attributes:
        knowledge_id: Unique identifier
        domain: Domain area (telecom, banking, etc.)
        category: Knowledge category
        fact: The actual knowledge content
        source: Where this knowledge came from
        confidence: Confidence in the fact (0-1)
        tags: Tags for filtering
    """
    knowledge_id: str
    domain: str
    category: str
    fact: str
    source: str
    confidence: float
    tags: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knowledge_id": self.knowledge_id,
            "domain": self.domain,
            "category": self.category,
            "fact": self.fact,
            "source": self.source,
            "confidence": self.confidence,
            "tags": self.tags or []
        }


class SemanticRetriever:
    """
    Semantic retriever for domain knowledge.

    Uses Weaviate vector search to find relevant knowledge
    for augmenting sub-agent reasoning.

    Retrieval strategies:
    1. get_relevant_knowledge: Based on customer context
    2. search: Direct semantic search
    3. get_by_category: All knowledge in a category
    4. get_playbooks: Success playbooks for scenarios
    """

    # Maximum items to return
    MAX_RESULTS = 20

    # Minimum confidence threshold
    MIN_CONFIDENCE = 0.5

    def __init__(
        self,
        tenant_id: str,
        domain: str = None
    ):
        """
        Initialize semantic retriever.

        Args:
            tenant_id: Tenant identifier
            domain: Optional domain filter (telecom, banking, etc.)
        """
        self.tenant_id = tenant_id
        self.domain = domain

        # Initialize Weaviate
        self.weaviate = None
        if WEAVIATE_AVAILABLE:
            try:
                self.weaviate = create_weaviate_client()
            except Exception as e:
                logger.warning("Weaviate not available", error=str(e))

        logger.info(
            "SemanticRetriever initialized",
            tenant_id=tenant_id,
            domain=domain,
            weaviate_available=self.weaviate is not None
        )

    async def get_relevant_knowledge(
        self,
        customer_id: str = None,
        context: Dict[str, Any] = None,
        categories: List[str] = None,
        limit: int = None
    ) -> Dict[str, Any]:
        """
        Get relevant knowledge for a customer context.

        Builds a semantic query from the context and retrieves
        relevant knowledge items.

        Args:
            customer_id: Customer ID for context
            context: Additional context (scores, segment, etc.)
            categories: Filter by categories
            limit: Maximum items to return

        Returns:
            Dict with organized knowledge
        """
        limit = limit or self.MAX_RESULTS
        context = context or {}

        # Build query from context
        query_parts = []

        if customer_id:
            query_parts.append(f"customer {customer_id}")

        # Add context-based queries
        if context.get("churn_risk", 0) > 0.5:
            query_parts.append("retention offers discounts")

        if context.get("propensity_score", 0) > 0.6:
            query_parts.append("upsell upgrade premium")

        if context.get("segment"):
            query_parts.append(context["segment"])

        if context.get("product"):
            query_parts.append(context["product"])

        # Default query if nothing specific
        if not query_parts:
            query_parts.append("customer service policies")

        query = " ".join(query_parts)

        # Search
        results = await self.search(query, categories=categories, limit=limit)

        # Organize by category
        by_category: Dict[str, List[Dict]] = {
            "product": [],
            "policy": [],
            "procedure": [],
            "faq": [],
            "playbook": []
        }

        for item in results:
            cat = item.get("category", "faq")
            if cat in by_category:
                by_category[cat].append(item)

        return {
            "customer_id": customer_id,
            "query_used": query,
            "total_results": len(results),
            "knowledge_by_category": by_category,
            "all_knowledge": results
        }

    async def search(
        self,
        query: str,
        categories: List[str] = None,
        tags: List[str] = None,
        min_confidence: float = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for knowledge using semantic similarity.

        Args:
            query: Search query
            categories: Filter by categories
            tags: Filter by tags
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of matching knowledge items
        """
        if not self.weaviate:
            logger.warning("Weaviate not available for search")
            return []

        limit = limit or self.MAX_RESULTS
        min_confidence = min_confidence or self.MIN_CONFIDENCE

        try:
            # Build where filter
            operands = [
                {"path": ["tenant_id"], "operator": "Equal", "valueText": self.tenant_id},
                {"path": ["confidence"], "operator": "GreaterThanEqual", "valueNumber": min_confidence}
            ]

            if self.domain:
                operands.append(
                    {"path": ["domain"], "operator": "Equal", "valueText": self.domain}
                )

            if categories:
                cat_operands = [
                    {"path": ["category"], "operator": "Equal", "valueText": cat}
                    for cat in categories
                ]
                if len(cat_operands) > 1:
                    operands.append({"operator": "Or", "operands": cat_operands})
                else:
                    operands.append(cat_operands[0])

            where_filter = {"operator": "And", "operands": operands}

            # Execute query
            result = self.weaviate.query.get(
                "SemanticKnowledge",
                ["knowledge_id", "domain", "category", "fact", "source",
                 "confidence", "tags"]
            ).with_near_text(
                {"concepts": [query]}
            ).with_where(
                where_filter
            ).with_limit(limit).do()

            knowledge = result.get("data", {}).get("Get", {}).get("SemanticKnowledge", [])

            # Filter by tags if specified
            if tags:
                knowledge = [
                    k for k in knowledge
                    if any(tag in (k.get("tags") or []) for tag in tags)
                ]

            return knowledge

        except Exception as e:
            logger.error("Semantic search failed", error=str(e))
            return []

    async def get_by_category(
        self,
        category: str,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get all knowledge in a category.

        Args:
            category: Category to filter (product, policy, procedure, faq, playbook)
            limit: Maximum results

        Returns:
            List of knowledge items
        """
        return await self.search("", categories=[category], limit=limit)

    async def get_playbooks(
        self,
        scenario: str = None,
        action_type: str = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get success playbooks for scenarios.

        Args:
            scenario: Scenario description (e.g., "high churn risk")
            action_type: Action type (retention, upsell, etc.)
            limit: Maximum results

        Returns:
            List of playbook knowledge items
        """
        query_parts = ["playbook success pattern"]

        if scenario:
            query_parts.append(scenario)

        if action_type:
            query_parts.append(action_type)

        query = " ".join(query_parts)

        return await self.search(
            query,
            categories=["playbook"],
            limit=limit
        )

    async def add_knowledge(
        self,
        fact: str,
        category: str,
        source: str = "manual",
        confidence: float = 0.9,
        tags: List[str] = None,
        domain: str = None
    ) -> str:
        """
        Add a new knowledge item.

        Args:
            fact: The knowledge content
            category: Category (product, policy, procedure, faq, playbook)
            source: Source of knowledge
            confidence: Confidence in the fact
            tags: Tags for filtering
            domain: Domain (uses default if not specified)

        Returns:
            Knowledge ID
        """
        if not self.weaviate:
            logger.warning("Weaviate not available")
            return None

        import uuid
        knowledge_id = str(uuid.uuid4())

        try:
            data = {
                "knowledge_id": knowledge_id,
                "domain": domain or self.domain or "general",
                "category": category,
                "fact": fact,
                "source": source,
                "confidence": confidence,
                "tenant_id": self.tenant_id,
                "tags": tags or []
            }

            self.weaviate.data_object.create(
                data,
                "SemanticKnowledge",
                uuid=knowledge_id
            )

            logger.info(
                "Knowledge added",
                knowledge_id=knowledge_id,
                category=category
            )

            return knowledge_id

        except Exception as e:
            logger.error("Failed to add knowledge", error=str(e))
            return None

    async def get_context_prompt(
        self,
        customer_id: str = None,
        context: Dict[str, Any] = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Get knowledge formatted as a context prompt.

        Retrieves relevant knowledge and formats it for inclusion
        in an LLM prompt.

        Args:
            customer_id: Customer ID
            context: Additional context
            max_tokens: Approximate max tokens for context

        Returns:
            Formatted context string
        """
        knowledge = await self.get_relevant_knowledge(
            customer_id=customer_id,
            context=context,
            limit=10
        )

        lines = ["RELEVANT DOMAIN KNOWLEDGE:", ""]

        for cat, items in knowledge.get("knowledge_by_category", {}).items():
            if items:
                lines.append(f"## {cat.upper()}")
                for item in items[:3]:  # Max 3 per category
                    lines.append(f"- {item.get('fact', '')}")
                lines.append("")

        # Truncate if too long (rough estimate)
        result = "\n".join(lines)
        if len(result) > max_tokens * 4:  # ~4 chars per token
            result = result[:max_tokens * 4] + "\n...(truncated)"

        return result
