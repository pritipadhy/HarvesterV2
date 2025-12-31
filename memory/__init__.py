"""
Harvester v2 Memory System

Implements cross-session memory for continuous learning:

Components:
    - EpisodicMemory: Cross-session memories stored in Weaviate
    - SemanticRetriever: RAG for domain knowledge retrieval
    - PlaybookSelector: Match situations to success playbooks
    - ContextManager: Sliding window + summarization for context management

Memory types:
    - Episodic: Past interactions, outcomes, preferences
    - Semantic: Domain knowledge, facts, procedures
    - Procedural: Playbooks, workflows, success patterns
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harvester_v2.memory.episodic_memory import EpisodicMemory
    from harvester_v2.memory.semantic_retriever import SemanticRetriever
    from harvester_v2.memory.playbook_selector import PlaybookSelector
    from harvester_v2.memory.context_manager import ContextManager
