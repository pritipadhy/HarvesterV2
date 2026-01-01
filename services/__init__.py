"""
Harvester v2 Services Layer

Provides unified data ingestion, identity resolution, and simplified Kafka
integration with only 5 topics (down from 40+).

Services:
    - IngestionService: Unified data ingestion (replaces 30+ harvesters)
    - IdentityService: Identity resolution and golden record management
    - KafkaService: Simplified Kafka integration (5 topics)

Topics (5 total):
    1. harvester.customer.signals - Customer events/signals
    2. harvester.customer.context - Customer context updates
    3. harvester.intelligence.requests - Pipeline trigger requests
    4. harvester.intelligence.results - Pipeline output results
    5. harvester.alerts.priority - Critical/urgent alerts
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harvester_v2.services.kafka_service import KafkaService, HarvesterTopics

__all__ = [
    "KafkaService",
    "HarvesterTopics",
]
