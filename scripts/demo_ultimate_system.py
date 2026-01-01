"""
Demo: THE ULTIMATE SYSTEM - Unified Customer Intelligence

This demo runs the full HarvesterV2 intelligence pipeline showing:
1. Deep Research (with reflection loop)
2. Opportunity Detection
3. Risk Signal Aggregation
4. Actionable Recommendations (NBA + Retention)

Usage:
    PYTHONPATH=. python scripts/demo_ultimate_system.py
"""

import asyncio
import json
import os
from datetime import datetime

# Set API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyDyUCWUgzKfjZya5IfrHw308DTHXArHT0c"
os.environ["TAVILY_API_KEY"] = "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"

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


async def run_demo():
    """Run THE ULTIMATE SYSTEM demo."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  THE ULTIMATE SYSTEM - Unified Customer Intelligence Demo".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    print(f"\nDemo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show demo customer
    print_section("DEMO CUSTOMER SCENARIO", DEMO_CUSTOMER["context"])

    print("\n[LOADING] Initializing THE ULTIMATE SYSTEM...")

    try:
        from harvester_v2.orchestrator.intelligence_graph import HarvesterV2Graph

        # Initialize the graph
        graph = HarvesterV2Graph(
            tenant_id="gigaclear",
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
        result = await graph.run(DEMO_CUSTOMER)
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


if __name__ == "__main__":
    result = asyncio.run(run_demo())
