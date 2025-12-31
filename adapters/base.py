"""
Base Data Adapter with StreamState
===================================

Ported from EnterpriseGPT (enterprise-gpt/src/adapter/database/base.py)

Provides abstract base class for data source adapters with:
- StreamState: Bookmark tracking for incremental sync
- BaseDataAdapter: Abstract interface for read/write operations
- Retry policies via Tenacity

Usage:
    from harvester_v2.adapters import BaseDataAdapter, StreamState

    class PostgresDataAdapter(BaseDataAdapter):
        async def read_stream(self, stream, entity_ids, state=None):
            # Use state.cursor to resume from last position
            ...

        def next_state(self, stream, record, state):
            # Update cursor after processing record
            return StreamState(cursor=record["id"])
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, AsyncGenerator, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from shared.logging import get_platform_logger

logger = get_platform_logger(__name__)


class StreamState(BaseModel):
    """
    Bookmark state for incremental data sync.

    Tracks the position in a data stream to enable resumable
    processing without re-reading previously processed records.

    Attributes:
        cursor: High-water mark (ID, timestamp, offset)
        last_modified: When this state was last updated
        offset: Numeric offset for pagination
        metadata: Additional state information
    """
    cursor: Any = None
    last_modified: Optional[datetime] = None
    offset: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def update(self, new_cursor: Any) -> "StreamState":
        """
        Return a new StreamState with updated cursor.

        Args:
            new_cursor: New cursor value

        Returns:
            New StreamState instance
        """
        return StreamState(
            cursor=new_cursor,
            last_modified=datetime.utcnow(),
            offset=self.offset + 1,
            metadata=self.metadata.copy()
        )


class BaseDataAdapter(ABC):
    """
    Base class for data source adapters.

    Provides abstract interface for reading and writing data
    from various sources (databases, APIs, files) with:
    - Incremental sync via StreamState
    - Retry policies for resilience
    - Batch operations

    Subclasses must implement:
    - read_stream: Async generator yielding records
    - next_state: Update bookmark after processing
    - write_batch: Write records back to source

    Example:
        class SalesforceAdapter(BaseDataAdapter):
            async def read_stream(self, stream, entity_ids, state=None):
                query = self._build_query(stream, entity_ids, state)
                async for record in self._execute_query(query):
                    yield record

            def next_state(self, stream, record, state):
                return StreamState(cursor=record["LastModifiedDate"])
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adapter with configuration.

        Args:
            config: Adapter-specific configuration dict
        """
        self.config = config
        self._connection = None

    @abstractmethod
    async def read_stream(
        self,
        stream: str,
        entity_ids: List[str],
        state: Optional[StreamState] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Yield records from a data stream.

        Reads data from the specified stream, optionally filtered
        by entity IDs and resuming from a previous state.

        Args:
            stream: Stream/table/object name
            entity_ids: Filter to these entity IDs
            state: Previous state for incremental sync

        Yields:
            Dict records from the stream
        """
        pass

    @abstractmethod
    def next_state(
        self,
        stream: str,
        record: Dict[str, Any],
        state: Optional[StreamState]
    ) -> StreamState:
        """
        Compute next state after processing a record.

        Called after each record is processed to update the
        bookmark position. Enables resumable processing.

        Args:
            stream: Stream name
            record: The record just processed
            state: Current state (or None)

        Returns:
            Updated StreamState with new cursor
        """
        pass

    @abstractmethod
    async def write_batch(
        self,
        stream: str,
        records: List[Dict[str, Any]]
    ) -> bool:
        """
        Write a batch of records to the stream.

        Args:
            stream: Target stream/table name
            records: List of records to write

        Returns:
            True if successful
        """
        pass

    async def connect(self) -> None:
        """
        Establish connection to data source.

        Override for sources requiring explicit connection setup.
        """
        pass

    async def disconnect(self) -> None:
        """
        Close connection to data source.

        Override for sources requiring cleanup.
        """
        pass

    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()

    @staticmethod
    def retry_policy():
        """
        Get Tenacity retry decorator for database operations.

        Returns:
            Retry decorator with exponential backoff
        """
        return retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=1, max=8),
            retry=retry_if_exception_type(Exception),
            before_sleep=lambda retry_state: logger.warning(
                "Retrying after error",
                attempt=retry_state.attempt_number,
                error=str(retry_state.outcome.exception()) if retry_state.outcome else None
            )
        )

    async def read_all(
        self,
        stream: str,
        entity_ids: List[str],
        state: Optional[StreamState] = None
    ) -> List[Dict[str, Any]]:
        """
        Read all records from stream (non-streaming).

        Convenience method that collects all records into a list.
        Use read_stream for large datasets.

        Args:
            stream: Stream name
            entity_ids: Filter to these entities
            state: Previous state

        Returns:
            List of all records
        """
        records = []
        async for record in self.read_stream(stream, entity_ids, state):
            records.append(record)
        return records

    def save_state(self, stream: str, state: StreamState) -> None:
        """
        Persist state for later resumption.

        Override to implement state persistence (file, database, etc.)

        Args:
            stream: Stream name
            state: State to save
        """
        logger.debug(f"State not persisted (override save_state): {stream}")

    def load_state(self, stream: str) -> Optional[StreamState]:
        """
        Load previously saved state.

        Override to implement state loading.

        Args:
            stream: Stream name

        Returns:
            Saved state or None
        """
        return None
