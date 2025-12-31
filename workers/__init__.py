"""
Prefect Background Workers for Harvester V2
============================================

Ported from EnterpriseGPT (enterprise-gpt/src/workers/)

Provides batch processing with:
- Parallel customer scoring
- NBA generation workflows
- Automatic retries
- Progress tracking

Usage:
    # Run batch scoring flow
    python -m harvester_v2.workers.scoring_worker

    # Or import and run programmatically
    from harvester_v2.workers import batch_scoring_flow
    await batch_scoring_flow(customer_ids, tenant_id="cisco_gvse")
"""

from .scoring_worker import batch_scoring_flow, score_single_customer

__all__ = ["batch_scoring_flow", "score_single_customer"]
