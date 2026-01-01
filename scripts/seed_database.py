"""
HarvesterV2 Database Seed Script
================================

Seeds the database with sample customer data for testing and demos.

Usage:
    # From project root
    PYTHONPATH=. python harvester_v2/scripts/seed_database.py

    # With options
    PYTHONPATH=. python harvester_v2/scripts/seed_database.py --tenant gigaclear --count 10

Customers seeded represent different risk scenarios:
    - High churn risk (competitor mentions, complaints)
    - Fraud risk (unusual patterns)
    - Upsell opportunity (high usage, premium segment)
    - Contract renewal (approaching end date)
    - Happy customer (high NPS, low support tickets)
"""

import asyncio
import argparse
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Database configuration
DATABASE_URL = os.getenv(
    "HARVESTER_DATABASE_URL",
    "postgresql://afs_user:harvester_test_2025@localhost:5432/afs_dev"
)
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


# Sample data generators
FIRST_NAMES = ["James", "Sarah", "Michael", "Emma", "David", "Olivia", "Robert", "Sophia", "William", "Ava"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson", "Anderson"]
PRODUCTS = ["broadband_100", "broadband_500", "broadband_1000", "broadband_ultra", "business_fibre"]
SEGMENTS = ["basic", "standard", "premium", "enterprise"]
INDUSTRIES = ["residential", "small_business", "telecommunications", "retail", "healthcare", "finance"]
COMPETITORS = ["Hyperoptic", "Virgin Media", "BT", "Sky", "TalkTalk", "Vodafone"]
CHANNELS = ["phone", "email", "chat", "web", "app"]
ISSUES = [
    "slow speeds during peak hours",
    "connection drops frequently",
    "billing dispute",
    "considering switching providers",
    "router issues",
    "installation delay",
    "price inquiry",
    "upgrade request",
    "contract end date query",
    "service cancellation request"
]


def generate_customer_id(index: int, tenant_id: str) -> str:
    """Generate unique customer ID."""
    return f"CUST-{tenant_id.upper()[:3]}-{index:04d}"


def generate_customer_profile(index: int, tenant_id: str, scenario: str = None) -> Dict[str, Any]:
    """
    Generate a customer profile for a specific scenario.

    Scenarios:
        - high_churn: High churn risk customer
        - fraud_risk: Potential fraud signals
        - upsell: Upsell opportunity
        - renewal: Contract renewal approaching
        - happy: Happy loyal customer
        - random: Random mix
    """
    customer_id = generate_customer_id(index, tenant_id)
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)

    # Base profile
    profile = {
        "customer_id": customer_id,
        "tenant_id": tenant_id,
        "name": f"{first_name} {last_name}",
        "email": f"{first_name.lower()}.{last_name.lower()}@example.com",
        "phone": f"+44 7{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(1000, 9999)}",
        "segment": random.choice(SEGMENTS),
        "industry": random.choice(INDUSTRIES),
        "product": random.choice(PRODUCTS),
        "monthly_revenue": round(random.uniform(29.99, 199.99), 2),
        "tenure_months": random.randint(1, 60),
        "contract_start_date": datetime.now() - timedelta(days=random.randint(30, 1800)),
        "contract_end_date": datetime.now() + timedelta(days=random.randint(-30, 365)),
        "nps_score": random.randint(1, 10),
        "churn_risk": round(random.uniform(0.1, 0.5), 2),
        "fraud_risk": round(random.uniform(0.0, 0.2), 2),
        "propensity_score": round(random.uniform(0.3, 0.8), 2),
        "is_active": True
    }

    # Scenario-specific adjustments
    if scenario == "high_churn":
        profile.update({
            "nps_score": random.randint(1, 4),  # Detractor
            "churn_risk": round(random.uniform(0.7, 0.95), 2),
            "context": {
                "churn_signals": random.sample(["speed_complaints", "competitor_inquiry", "billing_dispute", "service_cancellation_request"], 3),
                "competitor_mentions": random.sample(COMPETITORS, 2),
                "support_tickets_last_90d": random.randint(5, 15),
                "usage_trend": "declining",
                "average_speed_mbps": random.randint(100, 400),  # Below expected
                "recent_interactions": [
                    {
                        "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                        "channel": random.choice(CHANNELS),
                        "sentiment": "negative",
                        "issue": random.choice(ISSUES[:5])  # Negative issues
                    }
                    for i in range(3)
                ]
            }
        })

    elif scenario == "fraud_risk":
        profile.update({
            "fraud_risk": round(random.uniform(0.6, 0.9), 2),
            "context": {
                "fraud_signals": ["unusual_payment_pattern", "multiple_address_changes", "high_value_dispute"],
                "payment_failures_last_90d": random.randint(3, 8),
                "address_changes_last_year": random.randint(3, 6),
                "identity_verification_issues": True,
                "recent_disputes": [
                    {"amount": random.randint(100, 500), "date": (datetime.now() - timedelta(days=i*10)).strftime("%Y-%m-%d")}
                    for i in range(random.randint(2, 4))
                ]
            }
        })

    elif scenario == "upsell":
        profile.update({
            "segment": "premium",
            "nps_score": random.randint(8, 10),  # Promoter
            "propensity_score": round(random.uniform(0.7, 0.95), 2),
            "churn_risk": round(random.uniform(0.05, 0.2), 2),
            "context": {
                "upsell_signals": ["high_usage", "premium_segment", "long_tenure", "positive_feedback"],
                "usage_trend": "increasing",
                "average_speed_mbps": random.randint(800, 950),  # Near limit
                "data_usage_gb_monthly": random.randint(500, 1000),
                "upgrade_inquiries": random.randint(1, 3),
                "support_tickets_last_90d": random.randint(0, 2),
                "recent_interactions": [
                    {
                        "date": (datetime.now() - timedelta(days=i*7)).strftime("%Y-%m-%d"),
                        "channel": random.choice(CHANNELS),
                        "sentiment": "positive",
                        "issue": random.choice(["upgrade request", "price inquiry", "feature question"])
                    }
                    for i in range(2)
                ]
            }
        })

    elif scenario == "renewal":
        profile.update({
            "contract_end_date": datetime.now() + timedelta(days=random.randint(7, 45)),
            "context": {
                "renewal_signals": ["contract_ending_soon", "asked_about_contract_date"],
                "competitor_research": random.choice([True, False]),
                "loyalty_years": profile["tenure_months"] // 12,
                "lifetime_value": round(profile["monthly_revenue"] * profile["tenure_months"], 2),
                "recent_interactions": [
                    {
                        "date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                        "channel": "chat",
                        "sentiment": "neutral",
                        "issue": "asked about contract end date"
                    }
                ]
            }
        })

    elif scenario == "happy":
        profile.update({
            "nps_score": random.randint(9, 10),
            "churn_risk": round(random.uniform(0.01, 0.1), 2),
            "tenure_months": random.randint(24, 60),
            "context": {
                "loyalty_signals": ["long_tenure", "referral_program", "auto_pay_enrolled"],
                "referrals_made": random.randint(1, 5),
                "support_tickets_last_90d": 0,
                "usage_trend": "stable",
                "average_speed_mbps": random.randint(900, 1000),
                "recent_interactions": [
                    {
                        "date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                        "channel": "email",
                        "sentiment": "positive",
                        "issue": "thank you for great service"
                    }
                ]
            }
        })

    else:  # random
        profile["context"] = {
            "support_tickets_last_90d": random.randint(0, 5),
            "usage_trend": random.choice(["increasing", "stable", "declining"]),
            "average_speed_mbps": random.randint(300, 950),
        }

    return profile


def generate_signals(customer: Dict[str, Any], count: int = 5) -> List[Dict[str, Any]]:
    """Generate signals for a customer."""
    signals = []
    context = customer.get("context", {})

    # Generate signals based on customer context
    for i in range(count):
        signal_type = random.choice(["churn", "support", "usage", "billing", "feedback"])

        signal = {
            "customer_id": customer["customer_id"],
            "tenant_id": customer["tenant_id"],
            "signal_type": signal_type,
            "signal_name": f"{signal_type}_signal_{i}",
            "severity": random.choice(["low", "medium", "high"]),
            "value": round(random.uniform(0, 1), 2),
            "source": random.choice(["crm", "oss", "billing", "support"]),
            "channel": random.choice(CHANNELS),
            "event_time": datetime.now() - timedelta(days=random.randint(0, 30)),
            "metadata": {}
        }

        # Adjust based on customer scenario
        if context.get("churn_signals"):
            signal["severity"] = "high" if signal_type == "churn" else signal["severity"]

        signals.append(signal)

    return signals


async def create_tables(engine):
    """Create database tables."""
    from harvester_v2.storage.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("[OK] Database tables created")


async def seed_customers(session: AsyncSession, tenant_id: str, count: int) -> List[Dict[str, Any]]:
    """Seed customer data."""
    from harvester_v2.storage.models import CustomerProfile, CustomerSignal

    customers = []
    scenarios = ["high_churn", "fraud_risk", "upsell", "renewal", "happy", "random"]

    for i in range(1, count + 1):
        # Distribute scenarios
        scenario = scenarios[(i - 1) % len(scenarios)]

        profile_data = generate_customer_profile(i, tenant_id, scenario)

        # Extract context for JSONB field
        context = profile_data.pop("context", {})

        # Create customer profile
        customer = CustomerProfile(
            customer_id=profile_data["customer_id"],
            tenant_id=profile_data["tenant_id"],
            name=profile_data["name"],
            email=profile_data["email"],
            phone=profile_data["phone"],
            segment=profile_data["segment"],
            industry=profile_data["industry"],
            product=profile_data["product"],
            monthly_revenue=profile_data["monthly_revenue"],
            tenure_months=profile_data["tenure_months"],
            contract_start_date=profile_data["contract_start_date"],
            contract_end_date=profile_data["contract_end_date"],
            nps_score=profile_data["nps_score"],
            churn_risk=profile_data["churn_risk"],
            fraud_risk=profile_data["fraud_risk"],
            propensity_score=profile_data["propensity_score"],
            is_active=profile_data["is_active"],
            context=context
        )

        session.add(customer)
        customers.append({**profile_data, "context": context, "scenario": scenario})

        # Generate and add signals
        signals_data = generate_signals({**profile_data, "context": context}, count=random.randint(3, 8))
        for sig in signals_data:
            from harvester_v2.storage.models import SignalType, SignalSeverity
            signal = CustomerSignal(
                customer_id=sig["customer_id"],
                tenant_id=sig["tenant_id"],
                signal_type=SignalType(sig["signal_type"]),
                signal_name=sig["signal_name"],
                severity=SignalSeverity(sig["severity"]),
                value=sig["value"],
                source=sig["source"],
                channel=sig["channel"],
                event_time=sig["event_time"],
                signal_metadata=sig["metadata"]
            )
            session.add(signal)

    await session.commit()
    return customers


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Seed HarvesterV2 database")
    parser.add_argument("--tenant", default="gigaclear", help="Tenant ID")
    parser.add_argument("--count", type=int, default=10, help="Number of customers to seed")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate tables")
    args = parser.parse_args()

    print("=" * 60)
    print("  HarvesterV2 Database Seeder")
    print("=" * 60)
    print(f"\nTenant: {args.tenant}")
    print(f"Customer count: {args.count}")
    print(f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")

    # Create async engine
    engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)

    try:
        # Create tables
        print("\n[LOADING] Creating tables...")
        await create_tables(engine)

        # Create session
        from sqlalchemy.ext.asyncio import async_sessionmaker
        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session:
            # Seed customers
            print(f"\n[LOADING] Seeding {args.count} customers...")
            customers = await seed_customers(session, args.tenant, args.count)

            print(f"\n[OK] Seeded {len(customers)} customers")
            print("\nCustomers by scenario:")

            from collections import Counter
            scenario_counts = Counter(c["scenario"] for c in customers)
            for scenario, count in scenario_counts.items():
                print(f"  - {scenario}: {count}")

            # Print sample customers
            print("\nSample customers:")
            for customer in customers[:5]:
                print(f"  [{customer['scenario']:12}] {customer['customer_id']} - {customer['name']} "
                      f"(churn: {customer['churn_risk']:.2f}, NPS: {customer['nps_score']})")

    except Exception as e:
        print(f"\n[ERROR] Seeding failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        await engine.dispose()

    print("\n" + "=" * 60)
    print("  Seeding Complete!")
    print("=" * 60)
    print(f"\nTo run the pipeline with seeded data:")
    print(f"  PYTHONPATH=. python scripts/demo_ultimate_system.py --from-db")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
