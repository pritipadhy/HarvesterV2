"""
Harvester v2 Storage Layer

Database schemas and migrations for persistent storage:

Components:
    - weaviate_schema: Extended Weaviate schema for vector storage
        - EpisodicMemory: Cross-session memories
        - SemanticKnowledge: Domain knowledge for RAG
        - CustomerIntelligence: Unified customer intelligence
        - NextBestAction: Generated NBA recommendations

    - postgresql_schema: Learning tables
        - episodic_memories: Backup from Weaviate
        - playbooks: Procedural memory
        - action_outcomes: Reinforcement learning data
        - training_data: DPO pairs for fine-tuning
        - execution_traces: Observability data
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harvester_v2.storage.weaviate_schema import HARVESTER_V2_SCHEMA
