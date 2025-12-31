"""
Episodic Memory for Harvester V2
=================================

Manages cross-session memories for entities (customers, sessions, agents).
Stores memories in both Weaviate (for vector search) and PostgreSQL (for durability).

Memory types:
- interaction: Past conversation or action
- preference: Learned customer preference
- outcome: Result of an action taken
- insight: Strategic insight about the entity

Usage:
    memory = EpisodicMemory(tenant_id="gigaclear")
    await memory.store(
        entity_id="CUST-123",
        entity_type="customer",
        memory_type="interaction",
        content="Customer expressed interest in faster broadband",
        importance=0.8
    )

    memories = await memory.get_for_entity("CUST-123")
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import json
import structlog

from shared.logging import get_platform_logger

# Try to import database clients
try:
    from shared.database.weaviate_client import create_weaviate_client
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    from shared.database.postgresql_adapter import get_async_session
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

logger = get_platform_logger(__name__)


@dataclass
class Memory:
    """
    A single episodic memory.

    Attributes:
        memory_id: Unique identifier
        entity_id: ID of the entity this memory belongs to
        entity_type: Type of entity (customer, session, agent)
        memory_type: Type of memory (interaction, preference, outcome, insight)
        content: The actual memory content
        importance: Importance score (0-1) for retrieval prioritization
        context_summary: Summary of context when memory was created
        timestamp: When the memory was created
        metadata: Additional metadata
    """
    memory_id: str
    entity_id: str
    entity_type: str
    memory_type: str
    content: str
    importance: float
    context_summary: Optional[str] = None
    timestamp: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "memory_type": self.memory_type,
            "content": self.content,
            "importance": self.importance,
            "context_summary": self.context_summary,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata
        }


class EpisodicMemory:
    """
    Episodic memory manager for Harvester V2.

    Stores and retrieves cross-session memories with:
    - Vector search via Weaviate
    - Durable backup via PostgreSQL
    - Importance-based prioritization
    - Automatic expiration

    Memory retrieval strategies:
    1. get_for_entity: All memories for an entity
    2. search_similar: Semantic search across memories
    3. get_by_type: Memories of a specific type
    4. get_recent: Most recent memories
    """

    # Default TTL for memories (30 days)
    DEFAULT_TTL_DAYS = 30

    # Maximum memories to return
    MAX_MEMORIES = 50

    def __init__(
        self,
        tenant_id: str,
        ttl_days: int = None
    ):
        """
        Initialize episodic memory manager.

        Args:
            tenant_id: Tenant identifier
            ttl_days: Default TTL for memories (optional)
        """
        self.tenant_id = tenant_id
        self.ttl_days = ttl_days or self.DEFAULT_TTL_DAYS

        # Initialize clients
        self.weaviate = None
        if WEAVIATE_AVAILABLE:
            try:
                self.weaviate = create_weaviate_client()
            except Exception as e:
                logger.warning("Weaviate not available", error=str(e))

        logger.info(
            "EpisodicMemory initialized",
            tenant_id=tenant_id,
            weaviate_available=self.weaviate is not None
        )

    async def store(
        self,
        entity_id: str,
        entity_type: str,
        memory_type: str,
        content: str,
        importance: float = 0.5,
        context_summary: str = None,
        session_id: str = None,
        agent_name: str = None,
        ttl_days: int = None,
        metadata: Dict[str, Any] = None
    ) -> Memory:
        """
        Store a new episodic memory.

        Args:
            entity_id: ID of the entity
            entity_type: Type of entity (customer, session, agent)
            memory_type: Type of memory (interaction, preference, outcome, insight)
            content: The memory content
            importance: Importance score (0-1)
            context_summary: Summary of context
            session_id: Session ID if applicable
            agent_name: Agent that created the memory
            ttl_days: Custom TTL (optional)
            metadata: Additional metadata

        Returns:
            The created Memory object
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        expires_at = timestamp + timedelta(days=ttl_days or self.ttl_days)

        memory = Memory(
            memory_id=memory_id,
            entity_id=entity_id,
            entity_type=entity_type,
            memory_type=memory_type,
            content=content,
            importance=max(0.0, min(1.0, importance)),
            context_summary=context_summary,
            timestamp=timestamp,
            expires_at=expires_at,
            metadata=metadata or {}
        )

        # Store in Weaviate (for vector search)
        if self.weaviate:
            await self._store_weaviate(memory, session_id, agent_name)

        # Store in PostgreSQL (for durability)
        if POSTGRES_AVAILABLE:
            await self._store_postgres(memory, session_id, agent_name)

        logger.info(
            "Memory stored",
            memory_id=memory_id,
            entity_id=entity_id,
            memory_type=memory_type,
            importance=importance
        )

        return memory

    async def get_for_entity(
        self,
        entity_id: str,
        memory_types: List[str] = None,
        min_importance: float = 0.0,
        limit: int = None,
        use_importance_scoring: bool = True
    ) -> Dict[str, Any]:
        """
        Get all memories for an entity with importance-weighted scoring.

        Args:
            entity_id: Entity ID to get memories for
            memory_types: Filter by memory types (optional)
            min_importance: Minimum importance threshold
            limit: Maximum memories to return
            use_importance_scoring: Apply importance + recency scoring (default True)

        Returns:
            Dict with memories organized by type, sorted by combined score
        """
        limit = limit or self.MAX_MEMORIES

        memories = []

        # Try Weaviate first
        if self.weaviate:
            memories = await self._get_from_weaviate(
                entity_id, memory_types, min_importance, limit * 2  # Fetch extra for scoring
            )

        # Fallback to PostgreSQL
        if not memories and POSTGRES_AVAILABLE:
            memories = await self._get_from_postgres(
                entity_id, memory_types, min_importance, limit * 2
            )

        # Apply importance + recency scoring
        if use_importance_scoring and memories:
            scored_memories = []
            for memory in memories:
                combined_score = self._calculate_combined_score(memory)
                memory["combined_score"] = combined_score
                scored_memories.append((combined_score, memory))

            # Sort by combined score (highest first)
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            memories = [m for _, m in scored_memories[:limit]]

            logger.debug(
                "Applied importance scoring",
                entity_id=entity_id,
                total_memories=len(scored_memories),
                returned=len(memories)
            )

        # Organize by type
        by_type: Dict[str, List[Dict]] = {
            "interaction": [],
            "preference": [],
            "outcome": [],
            "insight": []
        }

        for memory in memories:
            mtype = memory.get("memory_type", "interaction")
            if mtype in by_type:
                by_type[mtype].append(memory)

        return {
            "entity_id": entity_id,
            "memory_count": len(memories),
            "memories_by_type": by_type,
            "memories": memories,  # Scored and sorted list
            "all_memories": memories  # Backwards compatibility
        }

    def _calculate_combined_score(
        self,
        memory: Dict[str, Any],
        importance_weight: float = 0.6,
        recency_weight: float = 0.4
    ) -> float:
        """
        Calculate combined importance + recency score.

        Score = (importance * weight) + (recency_score * weight)

        Recency decay:
        - Last 24h: 1.0
        - Last 7 days: 0.8
        - Last 30 days: 0.5
        - Older: 0.2

        Args:
            memory: Memory dict
            importance_weight: Weight for importance (default 0.6)
            recency_weight: Weight for recency (default 0.4)

        Returns:
            Combined score (0-1)
        """
        # Get importance score
        importance = memory.get("importance", 0.5)

        # Calculate recency score
        recency_score = 0.2  # Default for old memories

        timestamp = memory.get("timestamp")
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    # Handle ISO format with or without timezone
                    timestamp = timestamp.replace("Z", "+00:00")
                    # Remove fractional seconds if causing issues
                    if "." in timestamp:
                        parts = timestamp.split(".")
                        frac_and_tz = parts[1]
                        # Find timezone marker
                        if "+" in frac_and_tz:
                            frac, tz = frac_and_tz.split("+", 1)
                            timestamp = parts[0] + "." + frac[:6] + "+" + tz
                        elif "-" in frac_and_tz[1:]:  # Skip first char for negative offset
                            idx = frac_and_tz.rfind("-")
                            frac = frac_and_tz[:idx]
                            tz = frac_and_tz[idx+1:]
                            timestamp = parts[0] + "." + frac[:6] + "-" + tz
                    timestamp = datetime.fromisoformat(timestamp)

                # Calculate age
                now = datetime.utcnow()
                if timestamp.tzinfo:
                    now = now.replace(tzinfo=timestamp.tzinfo)

                age = now - timestamp
                age_hours = age.total_seconds() / 3600

                if age_hours <= 24:
                    recency_score = 1.0
                elif age_hours <= 24 * 7:  # 7 days
                    recency_score = 0.8
                elif age_hours <= 24 * 30:  # 30 days
                    recency_score = 0.5
                else:
                    recency_score = 0.2

            except Exception as e:
                logger.debug("Timestamp parsing failed, using default recency", error=str(e))
                recency_score = 0.3  # Fallback for parsing errors

        # Calculate combined score
        combined = (importance * importance_weight) + (recency_score * recency_weight)
        return min(1.0, max(0.0, combined))

    async def search_similar(
        self,
        query: str,
        entity_id: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar memories.

        Args:
            query: Search query
            entity_id: Optionally filter by entity
            limit: Maximum results

        Returns:
            List of similar memories with scores
        """
        if not self.weaviate:
            logger.warning("Weaviate not available for semantic search")
            return []

        try:
            # Build query
            query_obj = self.weaviate.query.get(
                "EpisodicMemory",
                ["entity_id", "entity_type", "memory_type", "content", "importance"]
            ).with_near_text({"concepts": [query]}).with_limit(limit)

            # Add filter for tenant and optional entity
            where_filter = {
                "operator": "And",
                "operands": [
                    {"path": ["tenant_id"], "operator": "Equal", "valueText": self.tenant_id}
                ]
            }

            if entity_id:
                where_filter["operands"].append(
                    {"path": ["entity_id"], "operator": "Equal", "valueText": entity_id}
                )

            query_obj = query_obj.with_where(where_filter)

            # Execute
            result = query_obj.do()
            memories = result.get("data", {}).get("Get", {}).get("EpisodicMemory", [])

            return memories

        except Exception as e:
            logger.error("Semantic search failed", error=str(e))
            return []

    async def get_recent(
        self,
        entity_id: str,
        hours: int = 24,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent memories for an entity.

        Args:
            entity_id: Entity ID
            hours: How far back to look
            limit: Maximum results

        Returns:
            List of recent memories
        """
        result = await self.get_for_entity(
            entity_id,
            limit=limit
        )

        # Filter by timestamp
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = []

        for memory in result.get("all_memories", []):
            timestamp = memory.get("timestamp")
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if timestamp >= cutoff:
                    recent.append(memory)

        return recent[:limit]

    async def delete_expired(self) -> int:
        """
        Delete expired memories.

        Returns:
            Number of memories deleted
        """
        deleted = 0

        if POSTGRES_AVAILABLE:
            try:
                async with get_async_session() as session:
                    from sqlalchemy import text
                    result = await session.execute(
                        text("""
                            DELETE FROM harvester_v2.episodic_memories
                            WHERE expires_at < NOW() AND tenant_id = :tenant_id
                            RETURNING id
                        """),
                        {"tenant_id": self.tenant_id}
                    )
                    deleted = result.rowcount
                    await session.commit()

                logger.info("Deleted expired memories", count=deleted)

            except Exception as e:
                logger.error("Failed to delete expired memories", error=str(e))

        return deleted

    # =========================================================================
    # Private Methods
    # =========================================================================

    async def _store_weaviate(
        self,
        memory: Memory,
        session_id: str = None,
        agent_name: str = None
    ) -> bool:
        """Store memory in Weaviate."""
        try:
            data = {
                "entity_id": memory.entity_id,
                "entity_type": memory.entity_type,
                "memory_type": memory.memory_type,
                "content": memory.content,
                "importance": memory.importance,
                "context_summary": memory.context_summary or "",
                "tenant_id": self.tenant_id,
                "session_id": session_id or "",
                "agent_name": agent_name or "",
                "timestamp": memory.timestamp.isoformat() if memory.timestamp else None,
                "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
                "metadata": json.dumps(memory.metadata or {})
            }

            self.weaviate.data_object.create(
                data,
                "EpisodicMemory",
                uuid=memory.memory_id
            )

            return True

        except Exception as e:
            logger.warning("Failed to store in Weaviate", error=str(e))
            return False

    async def _store_postgres(
        self,
        memory: Memory,
        session_id: str = None,
        agent_name: str = None
    ) -> bool:
        """Store memory in PostgreSQL."""
        try:
            async with get_async_session() as session:
                from sqlalchemy import text
                await session.execute(
                    text("""
                        INSERT INTO harvester_v2.episodic_memories
                        (id, entity_id, entity_type, memory_type, content, importance,
                         context_summary, tenant_id, session_id, agent_name,
                         created_at, expires_at, metadata)
                        VALUES
                        (:id, :entity_id, :entity_type, :memory_type, :content, :importance,
                         :context_summary, :tenant_id, :session_id, :agent_name,
                         :created_at, :expires_at, :metadata)
                        ON CONFLICT (entity_id, memory_type, tenant_id, created_at)
                        DO UPDATE SET content = :content, importance = :importance
                    """),
                    {
                        "id": memory.memory_id,
                        "entity_id": memory.entity_id,
                        "entity_type": memory.entity_type,
                        "memory_type": memory.memory_type,
                        "content": memory.content,
                        "importance": memory.importance,
                        "context_summary": memory.context_summary,
                        "tenant_id": self.tenant_id,
                        "session_id": session_id,
                        "agent_name": agent_name,
                        "created_at": memory.timestamp,
                        "expires_at": memory.expires_at,
                        "metadata": json.dumps(memory.metadata or {})
                    }
                )
                await session.commit()

            return True

        except Exception as e:
            logger.warning("Failed to store in PostgreSQL", error=str(e))
            return False

    async def _get_from_weaviate(
        self,
        entity_id: str,
        memory_types: List[str] = None,
        min_importance: float = 0.0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get memories from Weaviate."""
        try:
            # Build where filter
            operands = [
                {"path": ["tenant_id"], "operator": "Equal", "valueText": self.tenant_id},
                {"path": ["entity_id"], "operator": "Equal", "valueText": entity_id},
                {"path": ["importance"], "operator": "GreaterThanEqual", "valueNumber": min_importance}
            ]

            if memory_types:
                type_operands = [
                    {"path": ["memory_type"], "operator": "Equal", "valueText": mt}
                    for mt in memory_types
                ]
                if len(type_operands) > 1:
                    operands.append({"operator": "Or", "operands": type_operands})
                else:
                    operands.append(type_operands[0])

            where_filter = {"operator": "And", "operands": operands}

            result = self.weaviate.query.get(
                "EpisodicMemory",
                ["entity_id", "entity_type", "memory_type", "content",
                 "importance", "context_summary", "timestamp"]
            ).with_where(where_filter).with_limit(limit).do()

            memories = result.get("data", {}).get("Get", {}).get("EpisodicMemory", [])
            return memories

        except Exception as e:
            logger.warning("Failed to get from Weaviate", error=str(e))
            return []

    async def _get_from_postgres(
        self,
        entity_id: str,
        memory_types: List[str] = None,
        min_importance: float = 0.0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get memories from PostgreSQL."""
        try:
            async with get_async_session() as session:
                from sqlalchemy import text

                query = """
                    SELECT id, entity_id, entity_type, memory_type, content,
                           importance, context_summary, created_at as timestamp,
                           metadata
                    FROM harvester_v2.episodic_memories
                    WHERE tenant_id = :tenant_id
                      AND entity_id = :entity_id
                      AND importance >= :min_importance
                      AND (expires_at IS NULL OR expires_at > NOW())
                """

                if memory_types:
                    query += " AND memory_type = ANY(:memory_types)"

                query += " ORDER BY importance DESC, created_at DESC LIMIT :limit"

                params = {
                    "tenant_id": self.tenant_id,
                    "entity_id": entity_id,
                    "min_importance": min_importance,
                    "limit": limit
                }

                if memory_types:
                    params["memory_types"] = memory_types

                result = await session.execute(text(query), params)
                rows = result.fetchall()

                return [dict(row._mapping) for row in rows]

        except Exception as e:
            logger.warning("Failed to get from PostgreSQL", error=str(e))
            return []
