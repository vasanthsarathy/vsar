"""Trace collector for building explanation DAG."""

import uuid
from typing import Any

from vsar.trace.events import TraceEvent


class TraceCollector:
    """Collects trace events and builds an explanation DAG.

    The collector maintains a list of events where each event can reference
    parent events via their IDs, forming a directed acyclic graph (DAG).

    Example:
        >>> collector = TraceCollector()
        >>> event_id = collector.record("query", {"predicate": "parent"})
        >>> retrieval_id = collector.record(
        ...     "retrieval",
        ...     {"results": [("alice", 0.95)]},
        ...     parent_ids=[event_id]
        ... )
        >>> dag = collector.get_dag()
        >>> len(dag)
        2
    """

    def __init__(self) -> None:
        """Initialize empty trace collector."""
        self.events: list[TraceEvent] = []
        self._event_map: dict[str, TraceEvent] = {}

    def record(
        self,
        event_type: str,
        payload: dict[str, Any],
        parent_ids: list[str] | None = None,
    ) -> str:
        """Record a new trace event.

        Args:
            event_type: Type of event (query, unbind, cleanup, retrieval)
            payload: Event-specific data
            parent_ids: Optional list of parent event IDs

        Returns:
            ID of the created event

        Example:
            >>> collector = TraceCollector()
            >>> event_id = collector.record("query", {"predicate": "parent"})
        """
        event_id = str(uuid.uuid4())
        event = TraceEvent(
            id=event_id,
            type=event_type,
            payload=payload,
            parent_ids=parent_ids or [],
        )
        self.events.append(event)
        self._event_map[event_id] = event
        return event_id

    def get_dag(self) -> list[TraceEvent]:
        """Get all events in the trace DAG.

        Returns:
            List of all trace events in chronological order

        Example:
            >>> collector = TraceCollector()
            >>> collector.record("query", {"predicate": "parent"})
            >>> dag = collector.get_dag()
            >>> len(dag)
            1
        """
        return self.events.copy()

    def get_event(self, event_id: str) -> TraceEvent | None:
        """Get a specific event by ID.

        Args:
            event_id: Event identifier

        Returns:
            TraceEvent if found, None otherwise

        Example:
            >>> collector = TraceCollector()
            >>> event_id = collector.record("query", {})
            >>> event = collector.get_event(event_id)
            >>> event.type
            'query'
        """
        return self._event_map.get(event_id)

    def get_subgraph(self, event_id: str) -> list[TraceEvent]:
        """Get subgraph of all events leading to the specified event.

        This includes the event itself and all its ancestors in the DAG.

        Args:
            event_id: Root event ID for the subgraph

        Returns:
            List of events in the subgraph (ancestors + root)

        Example:
            >>> collector = TraceCollector()
            >>> e1 = collector.record("query", {})
            >>> e2 = collector.record("retrieval", {}, parent_ids=[e1])
            >>> subgraph = collector.get_subgraph(e2)
            >>> len(subgraph)
            2
        """
        visited: set[str] = set()
        result: list[TraceEvent] = []

        def visit(eid: str) -> None:
            if eid in visited:
                return
            event = self._event_map.get(eid)
            if event is None:
                return

            visited.add(eid)
            # Visit parents first (DFS)
            for parent_id in event.parent_ids:
                visit(parent_id)
            result.append(event)

        visit(event_id)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert trace DAG to dictionary for serialization.

        Returns:
            Dictionary with all events

        Example:
            >>> collector = TraceCollector()
            >>> collector.record("query", {"predicate": "parent"})
            >>> data = collector.to_dict()
            >>> "events" in data
            True
        """
        return {"events": [event.to_dict() for event in self.events]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceCollector":
        """Create TraceCollector from dictionary.

        Args:
            data: Dictionary with "events" list

        Returns:
            TraceCollector instance

        Example:
            >>> data = {"events": [{"id": "1", "type": "query", "payload": {}}]}
            >>> collector = TraceCollector.from_dict(data)
            >>> len(collector.get_dag())
            1
        """
        collector = cls()
        for event_data in data.get("events", []):
            event = TraceEvent.from_dict(event_data)
            collector.events.append(event)
            collector._event_map[event.id] = event
        return collector

    def clear(self) -> None:
        """Clear all events from the collector.

        Example:
            >>> collector = TraceCollector()
            >>> collector.record("query", {})
            >>> collector.clear()
            >>> len(collector.get_dag())
            0
        """
        self.events.clear()
        self._event_map.clear()
