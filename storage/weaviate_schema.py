"""
Weaviate Schema for Harvester V2 Intelligence Engine
=====================================================

Creates 4 core collections for the Claude-native intelligence engine:
1. EpisodicMemory - Cross-session memories for entities
2. SemanticKnowledge - Domain knowledge for RAG retrieval
3. CustomerIntelligenceV2 - Unified customer intelligence (v2)
4. NextBestActionV2 - Generated NBA recommendations (v2)

Architecture:
- Follows DeepAgents pattern for memory management
- Supports episodic + semantic memory retrieval
- Enables playbook-driven reasoning
- Tracks NBA outcomes for learning

Integration:
- EpisodicMemory stores past interactions per entity
- SemanticKnowledge provides domain-specific RAG
- CustomerIntelligenceV2 stores aggregated intelligence
- NextBestActionV2 stores generated actions with outcomes
"""

from typing import Dict, Any, Optional
import os
import structlog

from shared.database.weaviate_client import create_weaviate_client

logger = structlog.get_logger(__name__)


# Schema definition as a constant for programmatic access
HARVESTER_V2_SCHEMA = {
    "classes": [
        {
            "class": "EpisodicMemory",
            "description": "Cross-session memories for entities",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "entity_id", "dataType": ["text"]},
                {"name": "entity_type", "dataType": ["text"]},
                {"name": "memory_type", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]},
                {"name": "importance", "dataType": ["number"]},
                {"name": "tenant_id", "dataType": ["text"]},
                {"name": "timestamp", "dataType": ["date"]}
            ]
        },
        {
            "class": "SemanticKnowledge",
            "description": "Domain knowledge for RAG retrieval",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "domain", "dataType": ["text"]},
                {"name": "fact", "dataType": ["text"]},
                {"name": "source", "dataType": ["text"]},
                {"name": "confidence", "dataType": ["number"]},
                {"name": "tenant_id", "dataType": ["text"]}
            ]
        },
        {
            "class": "CustomerIntelligenceV2",
            "description": "Unified customer intelligence (v2)",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "customer_id", "dataType": ["text"]},
                {"name": "golden_record", "dataType": ["text"]},
                {"name": "fraud_score", "dataType": ["number"]},
                {"name": "churn_risk", "dataType": ["number"]},
                {"name": "propensity_score", "dataType": ["number"]},
                {"name": "segment", "dataType": ["text"]},
                {"name": "ltv", "dataType": ["number"]},
                {"name": "causal_factors", "dataType": ["text"]},
                {"name": "tenant_id", "dataType": ["text"]},
                {"name": "model_version", "dataType": ["text"]},
                {"name": "last_updated", "dataType": ["date"]}
            ]
        },
        {
            "class": "NextBestActionV2",
            "description": "Generated NBA recommendations (v2)",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "customer_id", "dataType": ["text"]},
                {"name": "action_type", "dataType": ["text"]},
                {"name": "expected_value", "dataType": ["number"]},
                {"name": "confidence", "dataType": ["number"]},
                {"name": "script", "dataType": ["text"]},
                {"name": "reasoning_trace", "dataType": ["text"]},
                {"name": "model_used", "dataType": ["text"]},
                {"name": "tenant_id", "dataType": ["text"]},
                {"name": "outcome", "dataType": ["text"]},
                {"name": "outcome_reward", "dataType": ["number"]},
                {"name": "created_at", "dataType": ["date"]}
            ]
        }
    ]
}


class HarvesterV2WeaviateSchema:
    """
    Weaviate Schema Manager for Harvester V2

    Creates and manages 4 core collections:
    - EpisodicMemory: Cross-session memories
    - SemanticKnowledge: Domain knowledge for RAG
    - CustomerIntelligenceV2: Unified customer intelligence
    - NextBestActionV2: Generated NBA recommendations

    Usage:
        schema = HarvesterV2WeaviateSchema()
        schema.connect()
        schema.create_all_schemas()
    """

    def __init__(
        self,
        url: str = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize Harvester V2 Weaviate Schema Manager

        Args:
            url: Weaviate instance URL (defaults to WEAVIATE_URL env var)
            api_key: Optional API key for authentication
        """
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.api_key = api_key
        self.client = None

        logger.info(
            "Initialized HarvesterV2WeaviateSchema",
            url=self.url
        )

    def connect(self) -> bool:
        """
        Establish Weaviate connection

        Returns:
            True if connection successful
        """
        try:
            self.client = create_weaviate_client(self.url, api_key=self.api_key)

            # Verify connection
            if not self.client.is_ready():
                raise ConnectionError("Weaviate is not ready")

            logger.info("Connected to Weaviate", url=self.url)
            return True

        except Exception as e:
            logger.error("Failed to connect to Weaviate", error=str(e))
            raise

    def disconnect(self):
        """Close Weaviate connection"""
        logger.info("Weaviate connection closed")

    def create_all_schemas(self) -> Dict[str, bool]:
        """
        Create all 4 Harvester V2 schemas

        Returns:
            Dict mapping collection name to creation status
        """
        results = {}

        results["EpisodicMemory"] = self.create_episodic_memory_schema()
        results["SemanticKnowledge"] = self.create_semantic_knowledge_schema()
        results["CustomerIntelligenceV2"] = self.create_customer_intelligence_schema()
        results["NextBestActionV2"] = self.create_nba_schema()

        logger.info("Created all Harvester V2 schemas", results=results)
        return results

    def create_episodic_memory_schema(self) -> bool:
        """
        Create EpisodicMemory collection for cross-session memories

        Properties:
        - entity_id: Unique identifier for the entity (customer, session, etc.)
        - entity_type: Type of entity (customer, session, agent)
        - memory_type: Type of memory (interaction, preference, outcome, insight)
        - content: The actual memory content (vectorized for semantic search)
        - importance: Importance score (0-1) for memory retrieval prioritization
        - tenant_id: Multi-tenant isolation
        - timestamp: When the memory was created

        Memory types:
        - interaction: Past conversation or action
        - preference: Learned customer preference
        - outcome: Result of an action taken
        - insight: Strategic insight about the entity

        Returns:
            True if successful, False if already exists
        """
        try:
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]

            if "EpisodicMemory" in existing_classes:
                logger.info("EpisodicMemory schema already exists")
                return False

            episodic_memory_schema = {
                "class": "EpisodicMemory",
                "description": "Cross-session memories for entities (DeepAgents episodic memory)",
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "poolingStrategy": "masked_mean",
                        "vectorizeClassName": False
                    }
                },
                "properties": [
                    {
                        "name": "entity_id",
                        "dataType": ["string"],
                        "description": "Unique entity identifier (customer_id, session_id, etc.)",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "entity_type",
                        "dataType": ["string"],
                        "description": "Entity type: customer, session, agent, workflow",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "memory_type",
                        "dataType": ["string"],
                        "description": "Memory type: interaction, preference, outcome, insight",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Memory content (vectorized for semantic retrieval)"
                    },
                    {
                        "name": "importance",
                        "dataType": ["number"],
                        "description": "Importance score (0-1) for retrieval prioritization"
                    },
                    {
                        "name": "context_summary",
                        "dataType": ["text"],
                        "description": "Summary of the context when memory was created"
                    },
                    {
                        "name": "tenant_id",
                        "dataType": ["string"],
                        "description": "Tenant identifier for multi-tenant isolation",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "session_id",
                        "dataType": ["string"],
                        "description": "Session ID when memory was created",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "agent_name",
                        "dataType": ["string"],
                        "description": "Agent that created the memory",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "timestamp",
                        "dataType": ["date"],
                        "description": "Memory creation timestamp"
                    },
                    {
                        "name": "expires_at",
                        "dataType": ["date"],
                        "description": "Memory expiration timestamp (for TTL)"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional JSON metadata"
                    }
                ]
            }

            self.client.schema.create_class(episodic_memory_schema)
            logger.info("Created EpisodicMemory schema in Weaviate")
            return True

        except Exception as e:
            logger.error("Failed to create EpisodicMemory schema", error=str(e))
            raise

    def create_semantic_knowledge_schema(self) -> bool:
        """
        Create SemanticKnowledge collection for domain knowledge RAG

        Properties:
        - domain: Domain area (telecom, banking, insurance, etc.)
        - category: Knowledge category (product, policy, procedure, faq)
        - fact: The knowledge fact (vectorized for semantic search)
        - source: Source of the knowledge (manual, harvested, inferred)
        - confidence: Confidence score (0-1) in the fact
        - tenant_id: Multi-tenant isolation

        Categories:
        - product: Product information, features, pricing
        - policy: Business policies, rules, constraints
        - procedure: Standard operating procedures
        - faq: Frequently asked questions and answers
        - playbook: Success playbooks for specific scenarios

        Returns:
            True if successful, False if already exists
        """
        try:
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]

            if "SemanticKnowledge" in existing_classes:
                logger.info("SemanticKnowledge schema already exists")
                return False

            semantic_knowledge_schema = {
                "class": "SemanticKnowledge",
                "description": "Domain knowledge for RAG retrieval (DeepAgents semantic memory)",
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "poolingStrategy": "masked_mean",
                        "vectorizeClassName": False
                    }
                },
                "properties": [
                    {
                        "name": "knowledge_id",
                        "dataType": ["string"],
                        "description": "Unique knowledge identifier",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "domain",
                        "dataType": ["string"],
                        "description": "Domain area: telecom, banking, insurance, healthcare",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "category",
                        "dataType": ["string"],
                        "description": "Knowledge category: product, policy, procedure, faq, playbook",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "fact",
                        "dataType": ["text"],
                        "description": "The knowledge fact (vectorized for semantic retrieval)"
                    },
                    {
                        "name": "source",
                        "dataType": ["string"],
                        "description": "Source: manual, harvested, inferred, external_api",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "source_url",
                        "dataType": ["string"],
                        "description": "URL or reference to the source",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "confidence",
                        "dataType": ["number"],
                        "description": "Confidence score (0-1) in the fact"
                    },
                    {
                        "name": "version",
                        "dataType": ["int"],
                        "description": "Knowledge version (for updates)"
                    },
                    {
                        "name": "tenant_id",
                        "dataType": ["string"],
                        "description": "Tenant identifier for multi-tenant isolation",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "tags",
                        "dataType": ["text[]"],
                        "description": "Tags for filtering (e.g., 'retention', 'upsell', 'support')"
                    },
                    {
                        "name": "created_at",
                        "dataType": ["date"],
                        "description": "Creation timestamp"
                    },
                    {
                        "name": "updated_at",
                        "dataType": ["date"],
                        "description": "Last update timestamp"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional JSON metadata"
                    }
                ]
            }

            self.client.schema.create_class(semantic_knowledge_schema)
            logger.info("Created SemanticKnowledge schema in Weaviate")
            return True

        except Exception as e:
            logger.error("Failed to create SemanticKnowledge schema", error=str(e))
            raise

    def create_customer_intelligence_schema(self) -> bool:
        """
        Create CustomerIntelligenceV2 collection for unified customer intelligence

        Properties:
        - customer_id: Unique customer identifier
        - golden_record: JSON string of unified customer data
        - fraud_score: Fraud risk score (0-1)
        - churn_risk: Churn probability (0-1)
        - propensity_score: Upsell/cross-sell propensity (0-1)
        - segment: Customer segment (high_value, at_risk, growth, etc.)
        - ltv: Customer lifetime value (monetary)
        - causal_factors: JSON string of causal analysis results
        - tenant_id: Multi-tenant isolation
        - model_version: Version of the model that generated scores
        - last_updated: Last update timestamp

        Segments:
        - high_value: High LTV, low churn risk
        - at_risk: High churn risk
        - growth: High propensity, room for expansion
        - nurture: New customer, needs engagement
        - watch: High fraud risk, needs monitoring

        Returns:
            True if successful, False if already exists
        """
        try:
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]

            if "CustomerIntelligenceV2" in existing_classes:
                logger.info("CustomerIntelligenceV2 schema already exists")
                return False

            customer_intelligence_schema = {
                "class": "CustomerIntelligenceV2",
                "description": "Unified customer intelligence from Harvester V2 sub-agents",
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "poolingStrategy": "masked_mean",
                        "vectorizeClassName": False
                    }
                },
                "properties": [
                    {
                        "name": "customer_id",
                        "dataType": ["string"],
                        "description": "Unique customer identifier",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "golden_record",
                        "dataType": ["text"],
                        "description": "JSON string of unified customer data"
                    },
                    {
                        "name": "fraud_score",
                        "dataType": ["number"],
                        "description": "Fraud risk score (0-1, higher = more risk)"
                    },
                    {
                        "name": "churn_risk",
                        "dataType": ["number"],
                        "description": "Churn probability (0-1, higher = more likely)"
                    },
                    {
                        "name": "propensity_score",
                        "dataType": ["number"],
                        "description": "Upsell/cross-sell propensity (0-1)"
                    },
                    {
                        "name": "risk_score",
                        "dataType": ["number"],
                        "description": "Composite risk score (0-1)"
                    },
                    {
                        "name": "segment",
                        "dataType": ["string"],
                        "description": "Customer segment: high_value, at_risk, growth, nurture, watch",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "ltv",
                        "dataType": ["number"],
                        "description": "Customer lifetime value (monetary)"
                    },
                    {
                        "name": "engagement_score",
                        "dataType": ["number"],
                        "description": "Customer engagement level (0-1)"
                    },
                    {
                        "name": "causal_factors",
                        "dataType": ["text"],
                        "description": "JSON string of causal analysis results"
                    },
                    {
                        "name": "reasoning_summary",
                        "dataType": ["text"],
                        "description": "Summary of reasoning that led to scores"
                    },
                    {
                        "name": "strategic_insights",
                        "dataType": ["text"],
                        "description": "JSON string of strategic insights"
                    },
                    {
                        "name": "tenant_id",
                        "dataType": ["string"],
                        "description": "Tenant identifier for multi-tenant isolation",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "model_version",
                        "dataType": ["string"],
                        "description": "Version of the model that generated scores",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "orchestrator_run_id",
                        "dataType": ["string"],
                        "description": "ID of the orchestrator run that generated this",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "created_at",
                        "dataType": ["date"],
                        "description": "Creation timestamp"
                    },
                    {
                        "name": "last_updated",
                        "dataType": ["date"],
                        "description": "Last update timestamp"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional JSON metadata"
                    },
                    # Cross-reference to episodic memories
                    {
                        "name": "related_memories",
                        "dataType": ["EpisodicMemory"],
                        "description": "Related episodic memories for this customer"
                    }
                ]
            }

            self.client.schema.create_class(customer_intelligence_schema)
            logger.info("Created CustomerIntelligenceV2 schema in Weaviate")
            return True

        except Exception as e:
            logger.error("Failed to create CustomerIntelligenceV2 schema", error=str(e))
            raise

    def create_nba_schema(self) -> bool:
        """
        Create NextBestActionV2 collection for generated NBA recommendations

        Properties:
        - customer_id: Customer this action is for
        - action_type: Type of action (retention, upsell, cross_sell, service, escalation)
        - action_name: Specific action name
        - expected_value: Expected monetary impact
        - confidence: Confidence in the recommendation (0-1)
        - urgency: Action urgency (immediate, 24h, 7d, 30d)
        - script: Personalized talk track for the action
        - reasoning_trace: Explanation of why this action was chosen
        - model_used: Model that generated this action
        - tenant_id: Multi-tenant isolation
        - outcome: Actual outcome (pending, accepted, rejected, ignored)
        - outcome_reward: Reward signal from outcome
        - created_at: Creation timestamp

        Returns:
            True if successful, False if already exists
        """
        try:
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]

            if "NextBestActionV2" in existing_classes:
                logger.info("NextBestActionV2 schema already exists")
                return False

            nba_schema = {
                "class": "NextBestActionV2",
                "description": "Generated NBA recommendations with outcomes for learning",
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "poolingStrategy": "masked_mean",
                        "vectorizeClassName": False
                    }
                },
                "properties": [
                    {
                        "name": "nba_id",
                        "dataType": ["string"],
                        "description": "Unique NBA identifier",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "customer_id",
                        "dataType": ["string"],
                        "description": "Customer this action is for",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "action_type",
                        "dataType": ["string"],
                        "description": "Action type: retention, upsell, cross_sell, service, escalation",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "action_name",
                        "dataType": ["text"],
                        "description": "Specific action name (e.g., 'Offer 20% discount', 'Upgrade to premium')"
                    },
                    {
                        "name": "expected_value",
                        "dataType": ["number"],
                        "description": "Expected monetary impact of the action"
                    },
                    {
                        "name": "confidence",
                        "dataType": ["number"],
                        "description": "Confidence in the recommendation (0-1)"
                    },
                    {
                        "name": "urgency",
                        "dataType": ["string"],
                        "description": "Action urgency: immediate, 24h, 7d, 30d",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "script",
                        "dataType": ["text"],
                        "description": "Personalized talk track for the action"
                    },
                    {
                        "name": "script_objections",
                        "dataType": ["text"],
                        "description": "JSON array of objection handlers"
                    },
                    {
                        "name": "reasoning_trace",
                        "dataType": ["text"],
                        "description": "Explanation of why this action was chosen"
                    },
                    {
                        "name": "causal_justification",
                        "dataType": ["text"],
                        "description": "Causal reasoning for the action"
                    },
                    {
                        "name": "model_used",
                        "dataType": ["string"],
                        "description": "Model that generated this action",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "subagent_scores",
                        "dataType": ["text"],
                        "description": "JSON object with critique scores from sub-agents"
                    },
                    {
                        "name": "tenant_id",
                        "dataType": ["string"],
                        "description": "Tenant identifier for multi-tenant isolation",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "orchestrator_run_id",
                        "dataType": ["string"],
                        "description": "ID of the orchestrator run that generated this",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    # Outcome tracking for learning
                    {
                        "name": "outcome",
                        "dataType": ["string"],
                        "description": "Actual outcome: pending, accepted, rejected, ignored, partial",
                        "moduleConfig": {
                            "text2vec-transformers": {"skip": True}
                        }
                    },
                    {
                        "name": "outcome_details",
                        "dataType": ["text"],
                        "description": "Details about the outcome"
                    },
                    {
                        "name": "outcome_reward",
                        "dataType": ["number"],
                        "description": "Reward signal from outcome (for RL)"
                    },
                    {
                        "name": "actual_value",
                        "dataType": ["number"],
                        "description": "Actual monetary impact achieved"
                    },
                    {
                        "name": "outcome_timestamp",
                        "dataType": ["date"],
                        "description": "When the outcome was recorded"
                    },
                    # Timestamps
                    {
                        "name": "created_at",
                        "dataType": ["date"],
                        "description": "Creation timestamp"
                    },
                    {
                        "name": "expires_at",
                        "dataType": ["date"],
                        "description": "Action expiration timestamp"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Additional JSON metadata"
                    },
                    # Cross-references
                    {
                        "name": "for_customer",
                        "dataType": ["CustomerIntelligenceV2"],
                        "description": "Link to customer intelligence record"
                    }
                ]
            }

            self.client.schema.create_class(nba_schema)
            logger.info("Created NextBestActionV2 schema in Weaviate")
            return True

        except Exception as e:
            logger.error("Failed to create NextBestActionV2 schema", error=str(e))
            raise

    def verify_schemas(self) -> Dict[str, bool]:
        """
        Verify all 4 Harvester V2 schemas exist

        Returns:
            Dict mapping collection name to existence status
        """
        try:
            schema = self.client.schema.get()
            existing_classes = [c["class"] for c in schema.get("classes", [])]

            results = {
                "EpisodicMemory": "EpisodicMemory" in existing_classes,
                "SemanticKnowledge": "SemanticKnowledge" in existing_classes,
                "CustomerIntelligenceV2": "CustomerIntelligenceV2" in existing_classes,
                "NextBestActionV2": "NextBestActionV2" in existing_classes
            }

            logger.info("Verified Harvester V2 schemas", results=results)
            return results

        except Exception as e:
            logger.error("Failed to verify schemas", error=str(e))
            raise

    def delete_all_schemas(self) -> Dict[str, bool]:
        """
        Delete all 4 Harvester V2 schemas (USE WITH CAUTION!)

        Returns:
            Dict mapping collection name to deletion status
        """
        results = {}

        for collection in [
            "EpisodicMemory",
            "SemanticKnowledge",
            "CustomerIntelligenceV2",
            "NextBestActionV2"
        ]:
            try:
                self.client.schema.delete_class(collection)
                results[collection] = True
                logger.warning(f"Deleted {collection} schema")
            except Exception as e:
                results[collection] = False
                logger.error(f"Failed to delete {collection} schema", error=str(e))

        return results


# ==================== Convenience Functions ====================

def create_schema(client) -> Dict[str, bool]:
    """
    Create all Harvester V2 schemas using provided client.

    Convenience function for test setup scripts.

    Args:
        client: Weaviate client instance

    Returns:
        Dict mapping collection name to creation status
    """
    schema_mgr = HarvesterV2WeaviateSchema()
    schema_mgr.client = client
    return schema_mgr.create_all_schemas()


# ==================== Singleton Instance ====================

_harvester_v2_schema_instance: Optional[HarvesterV2WeaviateSchema] = None


def get_harvester_v2_schema() -> HarvesterV2WeaviateSchema:
    """
    Get HarvesterV2WeaviateSchema instance (singleton)

    Usage:
        schema_mgr = get_harvester_v2_schema()
        schema_mgr.connect()
        schema_mgr.create_all_schemas()
    """
    global _harvester_v2_schema_instance

    if _harvester_v2_schema_instance is None:
        _harvester_v2_schema_instance = HarvesterV2WeaviateSchema()

    return _harvester_v2_schema_instance


def close_harvester_v2_schema():
    """Close HarvesterV2WeaviateSchema connection"""
    global _harvester_v2_schema_instance

    if _harvester_v2_schema_instance:
        _harvester_v2_schema_instance.disconnect()
        _harvester_v2_schema_instance = None


# ==================== CLI for Testing ====================

if __name__ == "__main__":
    import sys

    # Command-line interface for creating schemas
    schema_mgr = get_harvester_v2_schema()
    schema_mgr.connect()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "create_all":
            results = schema_mgr.create_all_schemas()
            print(f"Created schemas: {results}")

        elif command == "create_episodic":
            result = schema_mgr.create_episodic_memory_schema()
            print(f"Created EpisodicMemory schema: {result}")

        elif command == "create_semantic":
            result = schema_mgr.create_semantic_knowledge_schema()
            print(f"Created SemanticKnowledge schema: {result}")

        elif command == "create_intelligence":
            result = schema_mgr.create_customer_intelligence_schema()
            print(f"Created CustomerIntelligenceV2 schema: {result}")

        elif command == "create_nba":
            result = schema_mgr.create_nba_schema()
            print(f"Created NextBestActionV2 schema: {result}")

        elif command == "verify":
            results = schema_mgr.verify_schemas()
            print(f"Schema verification: {results}")

        elif command == "delete_all":
            confirm = input("WARNING: DELETE ALL HARVESTER V2 SCHEMAS? Type 'yes' to confirm: ")
            if confirm.lower() == "yes":
                results = schema_mgr.delete_all_schemas()
                print(f"Deleted schemas: {results}")
            else:
                print("Aborted")

        else:
            print("Unknown command. Available: create_all, create_episodic, create_semantic, create_intelligence, create_nba, verify, delete_all")

    else:
        # Default: create all schemas
        results = schema_mgr.create_all_schemas()
        print(f"Created all schemas: {results}")

    schema_mgr.disconnect()
