"""
Data Adapters Module for Harvester V2
=====================================

Provides base classes and utilities for data source adapters:
- StreamState: Bookmark tracking for incremental sync
- BaseDataAdapter: Abstract base for all data adapters

Usage:
    from harvester_v2.adapters import BaseDataAdapter, StreamState

    class SalesforceAdapter(BaseDataAdapter):
        async def read_stream(self, stream, entity_ids, state=None):
            ...
"""

from .base import BaseDataAdapter, StreamState

__all__ = ["BaseDataAdapter", "StreamState"]
