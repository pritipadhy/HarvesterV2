"""
Harvester v2 Services Layer

Provides unified data ingestion, identity resolution, and simplified Kafka
integration with only 5 topics (down from 40+).

Services:
    - IngestionService: Unified data ingestion (replaces 30+ harvesters)
    - IdentityService: Identity resolution and golden record management
    - KafkaService: Simplified Kafka integration (5 topics)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harvester_v2.services.ingestion_service import IngestionService
    from harvester_v2.services.identity_service import IdentityService
    from harvester_v2.services.kafka_service import KafkaService
