"""
Prefect Batch Scoring Worker
=============================

Ported from EnterpriseGPT (enterprise-gpt/src/workers/enrich_worker.py)

Provides batch customer scoring with:
- Parallel task execution
- Automatic retries (3 attempts)
- Progress logging
- Result aggregation

Usage:
    # Run as script
    PYTHONPATH=. python -m harvester_v2.workers.scoring_worker

    # Or import and use
    from harvester_v2.workers import batch_scoring_flow
    results = await batch_scoring_flow(
        customer_ids=["ACC-101", "ACC-102"],
        tenant_id="cisco_gvse"
    )
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path

from shared.logging import get_platform_logger

logger = get_platform_logger(__name__)

# Try to import Prefect
try:
    from prefect import flow, task
    from prefect.task_runners import ConcurrentTaskRunner
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    logger.warning("Prefect not installed. Workers will use basic async execution.")

    # Fallback decorators
    def task(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def flow(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@task(retries=3, retry_delay_seconds=5, log_prints=True)
async def score_single_customer(
    customer_id: str,
    tenant_id: str,
    input_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Score a single customer through Harvester V2 pipeline.

    Runs the customer through all sub-agents:
    - Scoring (fraud, churn, propensity)
    - Causality analysis
    - Reasoning
    - NBA generation
    - Retainer strategy

    Args:
        customer_id: Customer identifier
        tenant_id: Tenant identifier
        input_data: Optional customer data (if not loading from DB)

    Returns:
        Dict with scoring results and NBAs
    """
    from harvester_v2.orchestrator.intelligence_graph import HarvesterV2Graph

    logger.info(
        f"Scoring customer",
        customer_id=customer_id,
        tenant_id=tenant_id
    )

    start_time = datetime.utcnow()

    try:
        graph = HarvesterV2Graph(tenant_id=tenant_id)

        result = await graph.run(
            customer_id=customer_id,
            input_context=input_data or {}
        )

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            f"Customer scored successfully",
            customer_id=customer_id,
            execution_time_s=execution_time
        )

        return {
            "customer_id": customer_id,
            "tenant_id": tenant_id,
            "success": True,
            "result": result,
            "execution_time_s": execution_time,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        logger.error(
            f"Customer scoring failed",
            customer_id=customer_id,
            error=str(e)
        )

        return {
            "customer_id": customer_id,
            "tenant_id": tenant_id,
            "success": False,
            "error": str(e),
            "execution_time_s": execution_time,
            "timestamp": datetime.utcnow().isoformat()
        }


if PREFECT_AVAILABLE:
    @flow(
        name="batch-scoring-flow",
        description="Process multiple customers through Harvester V2",
        task_runner=ConcurrentTaskRunner(max_workers=10)
    )
    async def batch_scoring_flow(
        customer_ids: List[str],
        tenant_id: str,
        batch_size: int = 10,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process customers in parallel batches.

        Args:
            customer_ids: List of customer IDs to process
            tenant_id: Tenant identifier
            batch_size: Number of concurrent tasks
            output_path: Optional path to save results JSON

        Returns:
            List of scoring results
        """
        logger.info(
            f"Starting batch scoring",
            total_customers=len(customer_ids),
            tenant_id=tenant_id,
            batch_size=batch_size
        )

        results = []
        total = len(customer_ids)

        for i in range(0, total, batch_size):
            batch = customer_ids[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches}")

            # Submit tasks in parallel
            futures = [
                score_single_customer.submit(cid, tenant_id)
                for cid in batch
            ]

            # Gather results
            batch_results = await asyncio.gather(
                *[f.result() for f in futures],
                return_exceptions=True
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({
                        "success": False,
                        "error": str(result)
                    })
                else:
                    results.append(result)

            # Log progress
            completed = min(i + batch_size, total)
            success_count = sum(1 for r in results if r.get("success", False))
            logger.info(
                f"Batch progress: {completed}/{total} ({success_count} successful)"
            )

        # Save results if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_path}")

        # Summary
        success_count = sum(1 for r in results if r.get("success", False))
        logger.info(
            f"Batch scoring complete",
            total=len(results),
            successful=success_count,
            failed=len(results) - success_count
        )

        return results

else:
    # Fallback without Prefect
    async def batch_scoring_flow(
        customer_ids: List[str],
        tenant_id: str,
        batch_size: int = 10,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fallback batch scoring without Prefect."""
        logger.info(
            f"Starting batch scoring (no Prefect)",
            total_customers=len(customer_ids),
            tenant_id=tenant_id
        )

        results = []

        for i in range(0, len(customer_ids), batch_size):
            batch = customer_ids[i:i + batch_size]
            tasks = [
                score_single_customer(cid, tenant_id)
                for cid in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({"success": False, "error": str(result)})
                else:
                    results.append(result)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w") as f:
                json.dump(results, f, indent=2, default=str)

        return results


async def main():
    """Entry point for running as script."""
    import sys

    # Example usage
    customer_ids = ["ACC-101", "ACC-102", "ACC-103"]
    tenant_id = "cisco_gvse"

    if len(sys.argv) > 1:
        tenant_id = sys.argv[1]

    if len(sys.argv) > 2:
        customer_ids = sys.argv[2].split(",")

    results = await batch_scoring_flow(
        customer_ids=customer_ids,
        tenant_id=tenant_id,
        output_path="output/batch_scoring_results.json"
    )

    print(f"\nProcessed {len(results)} customers")
    print(f"Successful: {sum(1 for r in results if r.get('success', False))}")


if __name__ == "__main__":
    asyncio.run(main())
