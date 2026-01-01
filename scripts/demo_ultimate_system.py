"""
Demo: THE ULTIMATE SYSTEM - Unified Customer Intelligence

This demo runs the full HarvesterV2 intelligence pipeline showing:
1. Deep Research (with reflection loop)
2. Opportunity Detection
3. Risk Signal Aggregation
4. Actionable Recommendations (NBA + Retention)

Usage:
    # Run with hardcoded demo data
    PYTHONPATH=. python scripts/demo_ultimate_system.py

    # Run with database-seeded data
    PYTHONPATH=. python scripts/demo_ultimate_system.py --from-db

    # Run with Kafka trigger
    PYTHONPATH=. python scripts/demo_ultimate_system.py --kafka

    # Seed database first, then run
    PYTHONPATH=. python scripts/demo_ultimate_system.py --seed --from-db
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime

# Set API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyDyUCWUgzKfjZya5IfrHw308DTHXArHT0c"
os.environ["TAVILY_API_KEY"] = "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"

# Database configuration
DATABASE_URL = os.getenv(
    "HARVESTER_DATABASE_URL",
    "postgresql://afs_user:harvester_test_2025@localhost:5432/afs_dev"
)
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Demo customer scenario
DEMO_CUSTOMER = {
    "customer_id": "CUST-DEMO-001",
    "tenant_id": "gigaclear",
    "context": {
        # Customer profile
        "industry": "telecommunications",
        "product": "broadband_1000",
        "monthly_revenue": 89.99,
        "tenure_months": 28,
        "customer_segment": "premium",

        # Risk signals
        "churn_signals": ["speed_complaints", "competitor_inquiry", "billing_dispute"],
        "competitor_mentions": ["Hyperoptic", "Virgin Media"],
        "support_tickets_last_90d": 7,
        "nps_score": 5,  # Detractor

        # Recent interactions
        "recent_interactions": [
            {"date": "2024-12-28", "channel": "phone", "sentiment": "negative", "issue": "slow speeds during peak hours"},
            {"date": "2024-12-30", "channel": "email", "sentiment": "negative", "issue": "considering switching to Hyperoptic"},
            {"date": "2024-12-31", "channel": "chat", "sentiment": "neutral", "issue": "asked about contract end date"}
        ],

        # Usage data
        "usage_trend": "declining",
        "average_speed_mbps": 450,  # Below promised 1000
        "contract_end_date": "2025-02-15"
    }
}


def print_section(title: str, content: any = None):
    """Print a formatted section."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    if content:
        if isinstance(content, dict) or isinstance(content, list):
            print(json.dumps(content, indent=2, default=str))
        else:
            print(content)


async def run_demo(customer_data: dict = None):
    """Run THE ULTIMATE SYSTEM demo."""
    # Use provided customer data or default
    customer = customer_data or DEMO_CUSTOMER

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  THE ULTIMATE SYSTEM - Unified Customer Intelligence Demo".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    print(f"\nDemo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show demo customer
    print_section("DEMO CUSTOMER SCENARIO", customer["context"])

    print("\n[LOADING] Initializing THE ULTIMATE SYSTEM...")

    try:
        from harvester_v2.orchestrator.intelligence_graph import HarvesterV2Graph

        # Initialize the graph
        graph = HarvesterV2Graph(
            tenant_id=customer.get("tenant_id", "gigaclear"),
            use_checkpointing=False,  # Skip for demo
            use_langfuse=False  # Skip for demo
        )

        print("[OK] HarvesterV2Graph initialized")
        print(f"  - Sub-agents loaded: {len(graph.executor.subagents) if graph.executor else 0}")

        print("\n[LOADING] Running intelligence pipeline...")
        print("  Phase 1: Scoring (churn, fraud, propensity, risk)")
        print("  Phase 2: Analysis (Causality + Reasoning + Research + Opportunity + Risk)")
        print("  Phase 3: Actions (NBA + Retention Strategy)")
        print("  Phase 4: Critique & Quality Control")

        # Run the pipeline
        start_time = datetime.now()
        result = await graph.run(customer)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        print(f"\n[OK] Pipeline completed in {execution_time:.2f} seconds")
        print(f"  - Stages completed: {len(result.get('stage_history', []))}")
        print(f"  - Critique score: {result.get('critique_score', 0):.2f}")
        print(f"  - Iterations: {result.get('iteration_count', 1)}")

        # Display results
        print_section("SCORING RESULTS", result.get("scoring_result"))

        print_section("CAUSALITY ANALYSIS", result.get("causality_result"))

        # Research with reflection
        research = result.get("research_result", {})
        if research:
            print_section("DEEP RESEARCH (with Reflection Loop)")
            print(f"  Queries executed: {len(research.get('search_queries', []))}")
            print(f"  Sources found: {len(research.get('source_urls', []))}")
            print(f"  Reflection iterations: {research.get('reflection_iterations', 0)}")
            print(f"\n  Search Queries:")
            for q in research.get("search_queries", [])[:5]:
                print(f"    - {q}")
            print(f"\n  Key Findings:")
            for f in research.get("key_findings", [])[:3]:
                print(f"    - {f}")
            print(f"\n  Synthesis: {research.get('research_synthesis', '')[:500]}...")

        # Opportunities
        opportunities = result.get("opportunity_result", {})
        if opportunities:
            print_section("OPPORTUNITIES DETECTED")
            print(f"  Opportunity Score: {opportunities.get('opportunity_score', 0):.2f}")
            print(f"\n  Opportunities:")
            for opp in opportunities.get("opportunities", [])[:5]:
                print(f"    [{opp.get('type', 'unknown').upper()}] {opp.get('name', 'N/A')}")
                print(f"      Potential: {opp.get('potential_revenue_impact', 'N/A')}")
                print(f"      Confidence: {opp.get('confidence', 0):.2f}")
                print(f"      Timing: {opp.get('timing', 'N/A')}")

        # Risk Signals
        risk_signals = result.get("risk_signal_result", {})
        if risk_signals:
            print_section("RISK SIGNALS AGGREGATED")
            print(f"  Aggregate Risk Score: {risk_signals.get('aggregate_risk_score', 0):.2f}")
            print(f"\n  Risk Signals:")
            for risk in risk_signals.get("risk_signals", [])[:5]:
                severity = risk.get("severity", "unknown").upper()
                emoji = "[!!!]" if severity == "CRITICAL" else "[!!]" if severity == "HIGH" else "[!]"
                print(f"    {emoji} [{risk.get('type', 'unknown').upper()}] {risk.get('name', 'N/A')}")
                print(f"       Severity: {severity}")
                print(f"       Urgency: {risk.get('urgency', 'N/A')}")
                print(f"       Mitigation: {risk.get('recommended_mitigation', 'N/A')[:100]}")

            # Priority alerts
            alerts = risk_signals.get("priority_alerts", [])
            if alerts:
                print(f"\n  [ALERT] PRIORITY ALERTS:")
                for alert in alerts:
                    print(f"    - [{alert.get('severity', 'N/A').upper()}] {alert.get('message', 'N/A')}")

        # NBA Result
        print_section("NEXT BEST ACTION", result.get("nba_result"))

        # Retention Strategy
        print_section("RETENTION STRATEGY", result.get("retention_result"))

        # Final Intelligence Summary
        final_intel = result.get("final_intelligence", {})
        if final_intel:
            print_section("FINAL UNIFIED INTELLIGENCE")
            print(json.dumps(final_intel, indent=2, default=str)[:3000])

        # Summary
        print("\n" + "#" * 70)
        print("#" + " " * 68 + "#")
        print("#" + "  DEMO COMPLETE - THE ULTIMATE SYSTEM".center(68) + "#")
        print("#" + " " * 68 + "#")
        print("#" * 70)

        print(f"""
Summary:
  [OK] Scoring: Churn risk, fraud score, propensity calculated
  [OK] Causality: Root cause analysis completed
  [OK] Research: {len(research.get('source_urls', []))} sources found, {research.get('reflection_iterations', 0)} reflection iterations
  [OK] Opportunities: {len(opportunities.get('opportunities', []))} opportunities identified
  [OK] Risk Signals: {len(risk_signals.get('risk_signals', []))} risk signals detected
  [OK] NBA: Next best action with script generated
  [OK] Retention: Phased strategy created

  Execution Time: {execution_time:.2f} seconds
  Quality Score: {result.get('critique_score', 0):.2f}
""")

        return result

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def load_customer_from_db(customer_id: str, tenant_id: str):
    """Load customer data from database."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy import select

    print(f"\n[LOADING] Loading customer {customer_id} from database...")

    engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        from harvester_v2.storage.models import CustomerProfile, CustomerSignal

        async with async_session() as session:
            # Load customer
            query = select(CustomerProfile).where(
                CustomerProfile.customer_id == customer_id,
                CustomerProfile.tenant_id == tenant_id
            )
            result = await session.execute(query)
            customer = result.scalar_one_or_none()

            if not customer:
                print(f"[ERROR] Customer {customer_id} not found in database")
                return None

            # Load recent signals
            signals_query = select(CustomerSignal).where(
                CustomerSignal.customer_id == customer_id
            ).order_by(CustomerSignal.event_time.desc()).limit(20)

            signals_result = await session.execute(signals_query)
            signals = signals_result.scalars().all()

            # Build customer data
            customer_data = customer.to_dict()
            customer_data["context"]["recent_signals"] = [s.to_dict() for s in signals]

            print(f"[OK] Loaded customer: {customer.name} ({customer.segment})")
            print(f"  - Churn risk: {customer.churn_risk:.2f}")
            print(f"  - NPS score: {customer.nps_score}")
            print(f"  - Signals loaded: {len(signals)}")

            return customer_data

    finally:
        await engine.dispose()


async def list_customers_from_db(tenant_id: str, limit: int = 10):
    """List available customers from database."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy import select

    engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        from harvester_v2.storage.models import CustomerProfile

        async with async_session() as session:
            query = select(CustomerProfile).where(
                CustomerProfile.tenant_id == tenant_id,
                CustomerProfile.is_active == True
            ).order_by(CustomerProfile.churn_risk.desc()).limit(limit)

            result = await session.execute(query)
            customers = result.scalars().all()

            return customers

    finally:
        await engine.dispose()


async def seed_and_run(tenant_id: str):
    """Seed database and run demo."""
    print("\n[LOADING] Seeding database...")

    # Import and run seeder
    from harvester_v2.scripts.seed_database import seed_customers, create_tables
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

    engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)

    await create_tables(engine)

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        customers = await seed_customers(session, tenant_id, 10)
        print(f"[OK] Seeded {len(customers)} customers")

    await engine.dispose()

    # Get first high-churn customer
    customers = await list_customers_from_db(tenant_id)
    if customers:
        customer_data = await load_customer_from_db(customers[0].customer_id, tenant_id)
        if customer_data:
            return await run_demo(customer_data)

    return None


async def run_with_kafka(customer_id: str, tenant_id: str):
    """Trigger intelligence request via Kafka."""
    print("\n[LOADING] Sending intelligence request via Kafka...")

    from harvester_v2.services.kafka_service import KafkaService

    kafka = KafkaService()
    started = await kafka.start()

    if not started:
        print("[ERROR] Kafka not available")
        return None

    try:
        request_id = await kafka.request_intelligence(
            customer_id=customer_id,
            tenant_id=tenant_id,
            priority="high",
            source="demo"
        )

        if request_id:
            print(f"[OK] Intelligence request published: {request_id}")
            print("\nNote: Start the intelligence worker to process this request:")
            print("  PYTHONPATH=. python harvester_v2/workers/intelligence_worker.py")
        else:
            print("[ERROR] Failed to publish request")

        return request_id

    finally:
        await kafka.stop()


async def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="THE ULTIMATE SYSTEM Demo")
    parser.add_argument("--from-db", action="store_true", help="Load customer from database")
    parser.add_argument("--customer-id", default=None, help="Specific customer ID to load")
    parser.add_argument("--tenant", default="gigaclear", help="Tenant ID")
    parser.add_argument("--seed", action="store_true", help="Seed database before running")
    parser.add_argument("--kafka", action="store_true", help="Trigger via Kafka instead of direct run")
    parser.add_argument("--list", action="store_true", help="List available customers")
    args = parser.parse_args()

    if args.list:
        # List customers from database
        print("\n" + "=" * 60)
        print("  Available Customers in Database")
        print("=" * 60)

        customers = await list_customers_from_db(args.tenant)
        if not customers:
            print("\nNo customers found. Run with --seed to add sample data.")
        else:
            print(f"\nTenant: {args.tenant}")
            print(f"{'ID':<20} {'Name':<25} {'Segment':<12} {'Churn Risk':<10}")
            print("-" * 70)
            for c in customers:
                print(f"{c.customer_id:<20} {c.name:<25} {c.segment:<12} {c.churn_risk:.2f}")

        return None

    if args.seed:
        # Seed and run
        return await seed_and_run(args.tenant)

    if args.from_db:
        # Load from database
        customer_id = args.customer_id

        if not customer_id:
            # Get highest churn risk customer
            customers = await list_customers_from_db(args.tenant, limit=1)
            if not customers:
                print("[ERROR] No customers in database. Run with --seed first.")
                return None
            customer_id = customers[0].customer_id

        customer_data = await load_customer_from_db(customer_id, args.tenant)

        if not customer_data:
            return None

        if args.kafka:
            return await run_with_kafka(customer_id, args.tenant)
        else:
            return await run_demo(customer_data)

    elif args.kafka:
        # Kafka with hardcoded customer
        return await run_with_kafka(DEMO_CUSTOMER["customer_id"], DEMO_CUSTOMER["tenant_id"])

    else:
        # Run with hardcoded demo data
        return await run_demo(DEMO_CUSTOMER)


if __name__ == "__main__":
    result = asyncio.run(main())
