"""
HarvesterV2 Kafka Service
=========================

Simplified Kafka integration with only 5 topics (down from 40+ in v1).

Topics:
    1. harvester.customer.signals - Customer events/signals
    2. harvester.customer.context - Customer context updates
    3. harvester.intelligence.requests - Pipeline trigger requests
    4. harvester.intelligence.results - Pipeline output results
    5. harvester.alerts.priority - Critical/urgent alerts

Usage:
    from harvester_v2.services.kafka_service import KafkaService

    kafka = KafkaService(bootstrap_servers="localhost:9092")
    await kafka.start()

    # Publish signal
    await kafka.publish_signal("CUST-001", {"type": "churn", "value": 0.85})

    # Publish intelligence request
    await kafka.request_intelligence("CUST-001", priority="high")

    # Subscribe to results
    async for result in kafka.consume_results():
        print(result)
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, AsyncIterator, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

from shared.logging import get_platform_logger

logger = get_platform_logger(__name__)


# Topic names
class HarvesterTopics(str, Enum):
    """HarvesterV2 Kafka topics."""
    CUSTOMER_SIGNALS = "harvester.customer.signals"
    CUSTOMER_CONTEXT = "harvester.customer.context"
    INTELLIGENCE_REQUESTS = "harvester.intelligence.requests"
    INTELLIGENCE_RESULTS = "harvester.intelligence.results"
    PRIORITY_ALERTS = "harvester.alerts.priority"


@dataclass
class KafkaMessage:
    """Standard message format for HarvesterV2 Kafka messages."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tenant_id: str = ""
    customer_id: str = ""
    event_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, data: str) -> "KafkaMessage":
        """Deserialize from JSON."""
        parsed = json.loads(data)
        return cls(**parsed)


@dataclass
class IntelligenceRequest:
    """Request to run intelligence pipeline on a customer."""
    customer_id: str
    tenant_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: str = "normal"  # low, normal, high, critical
    source: str = "api"  # api, scheduler, signal, manual
    requested_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    context_override: Dict[str, Any] = field(default_factory=dict)
    callback_url: Optional[str] = None

    def to_kafka_message(self) -> KafkaMessage:
        """Convert to Kafka message."""
        return KafkaMessage(
            tenant_id=self.tenant_id,
            customer_id=self.customer_id,
            event_type="intelligence_request",
            payload={
                "request_id": self.request_id,
                "priority": self.priority,
                "source": self.source,
                "context_override": self.context_override,
                "callback_url": self.callback_url
            }
        )


class KafkaService:
    """
    Simplified Kafka service for HarvesterV2.

    Provides:
    - Producer for publishing signals, requests, and results
    - Consumer for processing requests and signals
    - Topic management
    """

    def __init__(
        self,
        bootstrap_servers: str = None,
        group_id: str = "harvester-v2",
        auto_create_topics: bool = True
    ):
        """
        Initialize Kafka service.

        Args:
            bootstrap_servers: Kafka bootstrap servers (default from env)
            group_id: Consumer group ID
            auto_create_topics: Create topics if they don't exist
        """
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.group_id = group_id
        self.auto_create_topics = auto_create_topics

        self._producer = None
        self._consumer = None
        self._running = False
        self._aiokafka_available = False

        # Check if aiokafka is available
        try:
            import aiokafka
            self._aiokafka_available = True
        except ImportError:
            logger.warning(
                "aiokafka not installed, Kafka features disabled",
                install_cmd="pip install aiokafka"
            )

    async def start(self) -> bool:
        """Start Kafka producer and create topics if needed."""
        if not self._aiokafka_available:
            logger.warning("Kafka service not started (aiokafka not available)")
            return False

        try:
            from aiokafka import AIOKafkaProducer
            from aiokafka.admin import AIOKafkaAdminClient, NewTopic

            # Create topics if needed
            if self.auto_create_topics:
                await self._create_topics()

            # Start producer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: v.encode("utf-8") if isinstance(v, str) else json.dumps(v).encode("utf-8")
            )
            await self._producer.start()
            self._running = True

            logger.info(
                "Kafka service started",
                bootstrap_servers=self.bootstrap_servers,
                topics=[t.value for t in HarvesterTopics]
            )
            return True

        except Exception as e:
            logger.error("Failed to start Kafka service", error=str(e))
            return False

    async def stop(self):
        """Stop Kafka connections."""
        self._running = False

        if self._producer:
            await self._producer.stop()
            self._producer = None

        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

        logger.info("Kafka service stopped")

    async def _create_topics(self):
        """Create Kafka topics if they don't exist."""
        try:
            from aiokafka.admin import AIOKafkaAdminClient, NewTopic

            admin = AIOKafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
            await admin.start()

            # Define topics with configurations
            topic_configs = [
                NewTopic(
                    name=HarvesterTopics.CUSTOMER_SIGNALS.value,
                    num_partitions=6,
                    replication_factor=1,
                    topic_configs={"retention.ms": str(7 * 24 * 60 * 60 * 1000)}  # 7 days
                ),
                NewTopic(
                    name=HarvesterTopics.CUSTOMER_CONTEXT.value,
                    num_partitions=3,
                    replication_factor=1,
                    topic_configs={"retention.ms": str(30 * 24 * 60 * 60 * 1000)}  # 30 days
                ),
                NewTopic(
                    name=HarvesterTopics.INTELLIGENCE_REQUESTS.value,
                    num_partitions=3,
                    replication_factor=1,
                    topic_configs={"retention.ms": str(24 * 60 * 60 * 1000)}  # 1 day
                ),
                NewTopic(
                    name=HarvesterTopics.INTELLIGENCE_RESULTS.value,
                    num_partitions=3,
                    replication_factor=1,
                    topic_configs={"retention.ms": str(7 * 24 * 60 * 60 * 1000)}  # 7 days
                ),
                NewTopic(
                    name=HarvesterTopics.PRIORITY_ALERTS.value,
                    num_partitions=1,
                    replication_factor=1,
                    topic_configs={"retention.ms": str(24 * 60 * 60 * 1000)}  # 1 day
                ),
            ]

            try:
                await admin.create_topics(topic_configs)
                logger.info("Kafka topics created", topics=[t.name for t in topic_configs])
            except Exception as e:
                # Topics may already exist
                if "TopicAlreadyExistsError" not in str(e):
                    logger.warning("Topic creation issue", error=str(e))

            await admin.close()

        except Exception as e:
            logger.warning("Could not create topics", error=str(e))

    # -------------------------------------------------------------------------
    # Producer Methods
    # -------------------------------------------------------------------------

    async def publish_signal(
        self,
        customer_id: str,
        tenant_id: str,
        signal_type: str,
        signal_data: Dict[str, Any]
    ) -> bool:
        """
        Publish a customer signal.

        Args:
            customer_id: Customer ID
            tenant_id: Tenant ID
            signal_type: Type of signal (churn, fraud, support, etc.)
            signal_data: Signal payload

        Returns:
            True if published successfully
        """
        if not self._producer:
            logger.warning("Kafka producer not available")
            return False

        message = KafkaMessage(
            tenant_id=tenant_id,
            customer_id=customer_id,
            event_type=f"signal.{signal_type}",
            payload=signal_data
        )

        try:
            await self._producer.send_and_wait(
                HarvesterTopics.CUSTOMER_SIGNALS.value,
                message.to_json(),
                key=customer_id.encode("utf-8")
            )
            logger.debug(
                "Signal published",
                customer_id=customer_id,
                signal_type=signal_type
            )
            return True
        except Exception as e:
            logger.error("Failed to publish signal", error=str(e))
            return False

    async def publish_context_update(
        self,
        customer_id: str,
        tenant_id: str,
        context: Dict[str, Any]
    ) -> bool:
        """Publish customer context update."""
        if not self._producer:
            return False

        message = KafkaMessage(
            tenant_id=tenant_id,
            customer_id=customer_id,
            event_type="context.update",
            payload=context
        )

        try:
            await self._producer.send_and_wait(
                HarvesterTopics.CUSTOMER_CONTEXT.value,
                message.to_json(),
                key=customer_id.encode("utf-8")
            )
            return True
        except Exception as e:
            logger.error("Failed to publish context update", error=str(e))
            return False

    async def request_intelligence(
        self,
        customer_id: str,
        tenant_id: str,
        priority: str = "normal",
        source: str = "api",
        context_override: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Request intelligence analysis for a customer.

        Args:
            customer_id: Customer ID
            tenant_id: Tenant ID
            priority: Request priority (low, normal, high, critical)
            source: Request source
            context_override: Override context values

        Returns:
            Request ID if published successfully
        """
        if not self._producer:
            logger.warning("Kafka producer not available")
            return None

        request = IntelligenceRequest(
            customer_id=customer_id,
            tenant_id=tenant_id,
            priority=priority,
            source=source,
            context_override=context_override or {}
        )

        message = request.to_kafka_message()

        try:
            await self._producer.send_and_wait(
                HarvesterTopics.INTELLIGENCE_REQUESTS.value,
                message.to_json(),
                key=customer_id.encode("utf-8")
            )
            logger.info(
                "Intelligence request published",
                request_id=request.request_id,
                customer_id=customer_id,
                priority=priority
            )
            return request.request_id
        except Exception as e:
            logger.error("Failed to publish intelligence request", error=str(e))
            return None

    async def publish_result(
        self,
        customer_id: str,
        tenant_id: str,
        request_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """Publish intelligence result."""
        if not self._producer:
            return False

        message = KafkaMessage(
            tenant_id=tenant_id,
            customer_id=customer_id,
            event_type="intelligence.result",
            payload={
                "request_id": request_id,
                "result": result
            }
        )

        try:
            await self._producer.send_and_wait(
                HarvesterTopics.INTELLIGENCE_RESULTS.value,
                message.to_json(),
                key=customer_id.encode("utf-8")
            )
            return True
        except Exception as e:
            logger.error("Failed to publish result", error=str(e))
            return False

    async def publish_alert(
        self,
        customer_id: str,
        tenant_id: str,
        alert_type: str,
        severity: str,
        message_text: str,
        details: Dict[str, Any] = None
    ) -> bool:
        """
        Publish priority alert.

        Args:
            customer_id: Customer ID
            tenant_id: Tenant ID
            alert_type: Type of alert (churn_critical, fraud_detected, etc.)
            severity: Alert severity (high, critical)
            message_text: Human-readable alert message
            details: Additional alert details
        """
        if not self._producer:
            return False

        message = KafkaMessage(
            tenant_id=tenant_id,
            customer_id=customer_id,
            event_type=f"alert.{alert_type}",
            payload={
                "severity": severity,
                "message": message_text,
                "details": details or {}
            }
        )

        try:
            await self._producer.send_and_wait(
                HarvesterTopics.PRIORITY_ALERTS.value,
                message.to_json(),
                key=tenant_id.encode("utf-8")  # Partition by tenant for alerts
            )
            logger.warning(
                "Priority alert published",
                customer_id=customer_id,
                alert_type=alert_type,
                severity=severity
            )
            return True
        except Exception as e:
            logger.error("Failed to publish alert", error=str(e))
            return False

    # -------------------------------------------------------------------------
    # Consumer Methods
    # -------------------------------------------------------------------------

    async def consume_requests(
        self,
        handler: Callable[[IntelligenceRequest], Any],
        tenant_filter: str = None
    ):
        """
        Consume intelligence requests and call handler.

        Args:
            handler: Async function to handle each request
            tenant_filter: Optional tenant ID to filter requests
        """
        if not self._aiokafka_available:
            logger.error("Cannot consume - aiokafka not available")
            return

        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            HarvesterTopics.INTELLIGENCE_REQUESTS.value,
            bootstrap_servers=self.bootstrap_servers,
            group_id=f"{self.group_id}-requests",
            value_deserializer=lambda m: json.loads(m.decode("utf-8"))
        )

        await consumer.start()
        self._consumer = consumer

        try:
            async for msg in consumer:
                if not self._running:
                    break

                try:
                    data = msg.value
                    message = KafkaMessage(**data)

                    # Apply tenant filter
                    if tenant_filter and message.tenant_id != tenant_filter:
                        continue

                    # Create request object
                    request = IntelligenceRequest(
                        customer_id=message.customer_id,
                        tenant_id=message.tenant_id,
                        request_id=message.payload.get("request_id", message.message_id),
                        priority=message.payload.get("priority", "normal"),
                        source=message.payload.get("source", "kafka"),
                        context_override=message.payload.get("context_override", {})
                    )

                    # Call handler
                    await handler(request)

                except Exception as e:
                    logger.error(
                        "Error processing request message",
                        error=str(e),
                        offset=msg.offset
                    )

        finally:
            await consumer.stop()

    async def consume_signals(
        self,
        handler: Callable[[KafkaMessage], Any],
        tenant_filter: str = None
    ):
        """
        Consume customer signals and call handler.

        Args:
            handler: Async function to handle each signal
            tenant_filter: Optional tenant ID to filter signals
        """
        if not self._aiokafka_available:
            return

        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            HarvesterTopics.CUSTOMER_SIGNALS.value,
            bootstrap_servers=self.bootstrap_servers,
            group_id=f"{self.group_id}-signals",
            value_deserializer=lambda m: json.loads(m.decode("utf-8"))
        )

        await consumer.start()

        try:
            async for msg in consumer:
                if not self._running:
                    break

                try:
                    message = KafkaMessage(**msg.value)

                    if tenant_filter and message.tenant_id != tenant_filter:
                        continue

                    await handler(message)

                except Exception as e:
                    logger.error("Error processing signal", error=str(e))

        finally:
            await consumer.stop()


# Singleton instance
_kafka_service: Optional[KafkaService] = None


def get_kafka_service() -> KafkaService:
    """Get or create Kafka service singleton."""
    global _kafka_service
    if _kafka_service is None:
        _kafka_service = KafkaService()
    return _kafka_service
