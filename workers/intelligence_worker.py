"""
HarvesterV2 Intelligence Worker
===============================

Kafka consumer worker that processes intelligence requests and runs
the HarvesterV2 pipeline.

Features:
    - Consumes from harvester.intelligence.requests topic
    - Loads customer context from database
    - Runs HarvesterV2Graph pipeline
    - Publishes results to harvester.intelligence.results
    - Publishes priority alerts when detected
    - Stores results in database

Usage:
    # Start worker
    PYTHONPATH=. python harvester_v2/workers/intelligence_worker.py

    # With options
    PYTHONPATH=. python harvester_v2/workers/intelligence_worker.py --tenant gigaclear --concurrency 5

Architecture:
    Kafka (requests) --> Worker --> HarvesterV2Graph --> Database + Kafka (results)
"""

import asyncio
import argparse
import os
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select

from shared.logging import get_platform_logger

logger = get_platform_logger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "HARVESTER_DATABASE_URL",
    "postgresql://afs_user:harvester_test_2025@localhost:5432/afs_dev"
)
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


class IntelligenceWorker:
    """
    Worker that consumes intelligence requests and runs the pipeline.

    Flow:
    1. Consume request from Kafka
    2. Load customer from database
    3. Run HarvesterV2Graph
    4. Store result in database
    5. Publish result to Kafka
    6. Publish alerts if any critical risks detected
    """

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        max_concurrency: int = 5,
        use_langfuse: bool = True
    ):
        """
        Initialize worker.

        Args:
            tenant_id: Optional tenant filter (None = all tenants)
            max_concurrency: Maximum concurrent pipeline executions
            use_langfuse: Enable Langfuse tracing
        """
        self.tenant_id = tenant_id
        self.max_concurrency = max_concurrency
        self.use_langfuse = use_langfuse

        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._engine = None
        self._session_factory = None
        self._kafka_service = None
        self._graph = None

        # Stats
        self._stats = {
            "requests_processed": 0,
            "requests_failed": 0,
            "alerts_published": 0,
            "started_at": None
        }

    async def start(self):
        """Start the worker."""
        logger.info(
            "Starting Intelligence Worker",
            tenant_id=self.tenant_id or "all",
            max_concurrency=self.max_concurrency
        )

        self._stats["started_at"] = datetime.utcnow()
        self._running = True

        # Initialize database
        await self._init_database()

        # Initialize Kafka
        await self._init_kafka()

        # Initialize HarvesterV2 graph
        await self._init_graph()

        # Start consuming
        await self._consume_requests()

    async def stop(self):
        """Stop the worker gracefully."""
        logger.info("Stopping Intelligence Worker...")
        self._running = False

        if self._kafka_service:
            await self._kafka_service.stop()

        if self._engine:
            await self._engine.dispose()

        logger.info(
            "Intelligence Worker stopped",
            stats=self._stats
        )

    async def _init_database(self):
        """Initialize database connection."""
        self._engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        logger.info("Database connection initialized")

    async def _init_kafka(self):
        """Initialize Kafka service."""
        from harvester_v2.services.kafka_service import KafkaService

        self._kafka_service = KafkaService()
        started = await self._kafka_service.start()

        if not started:
            logger.warning("Kafka not available, running in database-only mode")

    async def _init_graph(self):
        """Initialize HarvesterV2 graph."""
        from harvester_v2.orchestrator.intelligence_graph import HarvesterV2Graph

        self._graph = HarvesterV2Graph(
            tenant_id=self.tenant_id or "default",
            use_langfuse=self.use_langfuse,
            use_checkpointing=False
        )
        logger.info("HarvesterV2Graph initialized")

    async def _consume_requests(self):
        """Main loop to consume and process requests."""
        if self._kafka_service and self._kafka_service._running:
            # Kafka mode
            logger.info("Starting Kafka consumer...")

            async def handle_request(request):
                await self._process_request(
                    customer_id=request.customer_id,
                    tenant_id=request.tenant_id,
                    request_id=request.request_id,
                    priority=request.priority,
                    context_override=request.context_override
                )

            await self._kafka_service.consume_requests(
                handler=handle_request,
                tenant_filter=self.tenant_id
            )
        else:
            # Polling mode (no Kafka)
            logger.info("Running in polling mode (no Kafka)")
            await self._poll_database_requests()

    async def _poll_database_requests(self):
        """Poll database for pending intelligence requests."""
        from harvester_v2.storage.models import CustomerProfile

        while self._running:
            try:
                async with self._session_factory() as session:
                    # Find customers that need analysis
                    # (last_analyzed_at is NULL or > 1 hour ago, and high churn risk)
                    query = select(CustomerProfile).where(
                        CustomerProfile.is_active == True,
                        CustomerProfile.churn_risk >= 0.5
                    ).order_by(
                        CustomerProfile.churn_risk.desc()
                    ).limit(10)

                    if self.tenant_id:
                        query = query.where(CustomerProfile.tenant_id == self.tenant_id)

                    result = await session.execute(query)
                    customers = result.scalars().all()

                    for customer in customers:
                        if not self._running:
                            break

                        # Check if already analyzed recently
                        if customer.last_analyzed_at:
                            hours_since = (datetime.utcnow() - customer.last_analyzed_at).total_seconds() / 3600
                            if hours_since < 1:
                                continue

                        await self._process_request(
                            customer_id=customer.customer_id,
                            tenant_id=customer.tenant_id,
                            request_id=f"poll-{customer.customer_id}-{datetime.utcnow().isoformat()}",
                            priority="normal",
                            context_override={}
                        )

            except Exception as e:
                logger.error("Error in polling loop", error=str(e))

            # Wait before next poll
            await asyncio.sleep(60)

    async def _process_request(
        self,
        customer_id: str,
        tenant_id: str,
        request_id: str,
        priority: str,
        context_override: Dict[str, Any]
    ):
        """
        Process a single intelligence request.

        Args:
            customer_id: Customer ID
            tenant_id: Tenant ID
            request_id: Unique request ID
            priority: Request priority
            context_override: Override context values
        """
        async with self._semaphore:
            logger.info(
                "Processing intelligence request",
                request_id=request_id,
                customer_id=customer_id,
                priority=priority
            )

            start_time = datetime.utcnow()

            try:
                # 1. Load customer from database
                customer_data = await self._load_customer(customer_id, tenant_id)

                if not customer_data:
                    logger.warning(
                        "Customer not found",
                        customer_id=customer_id,
                        tenant_id=tenant_id
                    )
                    self._stats["requests_failed"] += 1
                    return

                # Apply context overrides
                if context_override:
                    customer_data["context"].update(context_override)

                # 2. Run pipeline
                result = await self._graph.run(customer_data)

                execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                # 3. Store result in database
                await self._store_result(
                    customer_id=customer_id,
                    tenant_id=tenant_id,
                    request_id=request_id,
                    result=result,
                    execution_time_ms=execution_time_ms
                )

                # 4. Publish result to Kafka
                if self._kafka_service:
                    await self._kafka_service.publish_result(
                        customer_id=customer_id,
                        tenant_id=tenant_id,
                        request_id=request_id,
                        result=result
                    )

                # 5. Check for critical alerts
                await self._check_and_publish_alerts(
                    customer_id=customer_id,
                    tenant_id=tenant_id,
                    result=result
                )

                self._stats["requests_processed"] += 1

                logger.info(
                    "Intelligence request completed",
                    request_id=request_id,
                    customer_id=customer_id,
                    execution_time_ms=execution_time_ms,
                    critique_score=result.get("critique_score", 0)
                )

            except Exception as e:
                self._stats["requests_failed"] += 1
                logger.error(
                    "Intelligence request failed",
                    request_id=request_id,
                    customer_id=customer_id,
                    error=str(e)
                )

    async def _load_customer(
        self,
        customer_id: str,
        tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load customer data from database."""
        from harvester_v2.storage.models import CustomerProfile, CustomerSignal

        async with self._session_factory() as session:
            # Load customer profile
            query = select(CustomerProfile).where(
                CustomerProfile.customer_id == customer_id,
                CustomerProfile.tenant_id == tenant_id
            )
            result = await session.execute(query)
            customer = result.scalar_one_or_none()

            if not customer:
                return None

            # Load recent signals
            signals_query = select(CustomerSignal).where(
                CustomerSignal.customer_id == customer_id,
                CustomerSignal.tenant_id == tenant_id
            ).order_by(
                CustomerSignal.event_time.desc()
            ).limit(50)

            signals_result = await session.execute(signals_query)
            signals = signals_result.scalars().all()

            # Build customer data dict
            customer_data = customer.to_dict()

            # Add recent signals to context
            customer_data["context"]["recent_signals"] = [
                s.to_dict() for s in signals
            ]

            return customer_data

    async def _store_result(
        self,
        customer_id: str,
        tenant_id: str,
        request_id: str,
        result: Dict[str, Any],
        execution_time_ms: int
    ):
        """Store pipeline result in database."""
        from harvester_v2.storage.models import (
            IntelligenceResult, CustomerProfile, RecommendedAction, ActionStatus
        )

        async with self._session_factory() as session:
            # Create result record
            intel_result = IntelligenceResult(
                customer_id=customer_id,
                tenant_id=tenant_id,
                workflow_id=request_id,
                churn_risk=result.get("scoring_result", {}).get("churn_risk"),
                fraud_risk=result.get("scoring_result", {}).get("fraud_score"),
                propensity_score=result.get("scoring_result", {}).get("propensity_score"),
                opportunity_score=result.get("opportunity_result", {}).get("opportunity_score"),
                aggregate_risk_score=result.get("risk_signal_result", {}).get("aggregate_risk_score"),
                critique_score=result.get("critique_score"),
                scoring_result=result.get("scoring_result"),
                causality_result=result.get("causality_result"),
                research_result=result.get("research_result"),
                opportunity_result=result.get("opportunity_result"),
                risk_signal_result=result.get("risk_signal_result"),
                nba_result=result.get("nba_result"),
                retention_result=result.get("retention_result"),
                final_intelligence=result.get("final_intelligence"),
                execution_time_ms=execution_time_ms,
                iterations=result.get("iteration_count", 1),
                subagents_executed=result.get("stage_history", [])
            )
            session.add(intel_result)

            # Update customer with latest scores
            query = select(CustomerProfile).where(
                CustomerProfile.customer_id == customer_id,
                CustomerProfile.tenant_id == tenant_id
            )
            customer_result = await session.execute(query)
            customer = customer_result.scalar_one_or_none()

            if customer:
                scoring = result.get("scoring_result", {})
                customer.churn_risk = scoring.get("churn_risk", customer.churn_risk)
                customer.fraud_risk = scoring.get("fraud_score", customer.fraud_risk)
                customer.propensity_score = scoring.get("propensity_score", customer.propensity_score)
                customer.last_analyzed_at = datetime.utcnow()

            # Create action records from NBA result
            nba = result.get("nba_result", {})
            if nba.get("recommended_action"):
                action = RecommendedAction(
                    customer_id=customer_id,
                    tenant_id=tenant_id,
                    action_type="nba",
                    action_name=nba.get("recommended_action", "Unknown"),
                    description=nba.get("justification"),
                    priority=1 if nba.get("urgency") == "immediate" else 3,
                    urgency=nba.get("urgency"),
                    script=nba.get("agent_script"),
                    offer_details=nba.get("offer_details"),
                    status=ActionStatus.PENDING
                )
                session.add(action)

            await session.commit()

            logger.debug(
                "Result stored in database",
                customer_id=customer_id,
                workflow_id=request_id
            )

    async def _check_and_publish_alerts(
        self,
        customer_id: str,
        tenant_id: str,
        result: Dict[str, Any]
    ):
        """Check for critical conditions and publish alerts."""
        if not self._kafka_service:
            return

        # Check churn risk
        scoring = result.get("scoring_result", {})
        churn_risk = scoring.get("churn_risk", 0)

        if churn_risk >= 0.85:
            await self._kafka_service.publish_alert(
                customer_id=customer_id,
                tenant_id=tenant_id,
                alert_type="churn_critical",
                severity="critical",
                message_text=f"Customer {customer_id} has critical churn risk: {churn_risk:.2%}",
                details={
                    "churn_risk": churn_risk,
                    "root_cause": result.get("causality_result", {}).get("root_cause")
                }
            )
            self._stats["alerts_published"] += 1

        # Check fraud risk
        fraud_risk = scoring.get("fraud_score", 0)
        if fraud_risk >= 0.7:
            await self._kafka_service.publish_alert(
                customer_id=customer_id,
                tenant_id=tenant_id,
                alert_type="fraud_detected",
                severity="critical",
                message_text=f"Customer {customer_id} has high fraud risk: {fraud_risk:.2%}",
                details={"fraud_score": fraud_risk}
            )
            self._stats["alerts_published"] += 1

        # Check priority risk signals
        risk_signals = result.get("risk_signal_result", {})
        priority_alerts = risk_signals.get("priority_alerts", [])

        for alert in priority_alerts:
            if alert.get("severity") in ["critical", "high"]:
                await self._kafka_service.publish_alert(
                    customer_id=customer_id,
                    tenant_id=tenant_id,
                    alert_type=alert.get("type", "risk_signal"),
                    severity=alert.get("severity"),
                    message_text=alert.get("message"),
                    details=alert
                )
                self._stats["alerts_published"] += 1


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HarvesterV2 Intelligence Worker")
    parser.add_argument("--tenant", default=None, help="Tenant ID filter (default: all)")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent pipelines")
    parser.add_argument("--no-langfuse", action="store_true", help="Disable Langfuse tracing")
    args = parser.parse_args()

    print("=" * 60)
    print("  HarvesterV2 Intelligence Worker")
    print("=" * 60)
    print(f"\nTenant: {args.tenant or 'all'}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Langfuse: {'disabled' if args.no_langfuse else 'enabled'}")

    worker = IntelligenceWorker(
        tenant_id=args.tenant,
        max_concurrency=args.concurrency,
        use_langfuse=not args.no_langfuse
    )

    # Handle signals for graceful shutdown
    loop = asyncio.get_event_loop()

    def handle_signal():
        logger.info("Received shutdown signal")
        asyncio.create_task(worker.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await worker.start()
    except KeyboardInterrupt:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
