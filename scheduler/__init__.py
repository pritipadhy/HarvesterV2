"""
Harvester v2 Scheduler

Batch processing and scheduled intelligence generation:

Components:
    - BatchProcessor: Scheduled batch intelligence processing
        - Daily customer scoring refresh
        - Weekly deep reasoning analysis
        - Monthly playbook optimization
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harvester_v2.scheduler.batch_processor import BatchProcessor
