"""Tests for trace events."""

from time import time

import pytest
from vsar.trace.events import TraceEvent


class TestTraceEvent:
    """Test TraceEvent dataclass."""

    def test_create_event_minimal(self) -> None:
        """Test creating event with minimal fields."""
        event = TraceEvent(id="evt1", type="query", payload={"predicate": "parent"})

        assert event.id == "evt1"
        assert event.type == "query"
        assert event.payload == {"predicate": "parent"}
        assert event.parent_ids == []
        assert isinstance(event.timestamp, float)

    def test_create_event_with_parents(self) -> None:
        """Test creating event with parent IDs."""
        event = TraceEvent(
            id="evt2",
            type="retrieval",
            payload={"results": []},
            parent_ids=["evt1"],
        )

        assert event.id == "evt2"
        assert event.parent_ids == ["evt1"]

    def test_create_event_with_timestamp(self) -> None:
        """Test creating event with explicit timestamp."""
        ts = time()
        event = TraceEvent(
            id="evt1",
            type="query",
            payload={},
            timestamp=ts,
        )

        assert event.timestamp == ts

    def test_to_dict(self) -> None:
        """Test converting event to dictionary."""
        event = TraceEvent(
            id="evt1",
            type="query",
            payload={"predicate": "parent", "args": ["alice", None]},
            parent_ids=["evt0"],
            timestamp=123.456,
        )

        data = event.to_dict()
        assert data["id"] == "evt1"
        assert data["type"] == "query"
        assert data["payload"]["predicate"] == "parent"
        assert data["parent_ids"] == ["evt0"]
        assert data["timestamp"] == 123.456

    def test_from_dict(self) -> None:
        """Test creating event from dictionary."""
        data = {
            "id": "evt1",
            "type": "unbind",
            "payload": {"position": 0},
            "parent_ids": ["evt0"],
            "timestamp": 123.456,
        }

        event = TraceEvent.from_dict(data)
        assert event.id == "evt1"
        assert event.type == "unbind"
        assert event.payload == {"position": 0}
        assert event.parent_ids == ["evt0"]
        assert event.timestamp == 123.456

    def test_from_dict_minimal(self) -> None:
        """Test creating event from minimal dictionary."""
        data = {
            "id": "evt1",
            "type": "query",
            "payload": {},
        }

        event = TraceEvent.from_dict(data)
        assert event.id == "evt1"
        assert event.type == "query"
        assert event.parent_ids == []
        assert isinstance(event.timestamp, float)

    def test_roundtrip_serialization(self) -> None:
        """Test to_dict and from_dict roundtrip."""
        original = TraceEvent(
            id="evt1",
            type="cleanup",
            payload={"operation": "test"},
            parent_ids=["evt0", "evt1"],
        )

        data = original.to_dict()
        reconstructed = TraceEvent.from_dict(data)

        assert reconstructed.id == original.id
        assert reconstructed.type == original.type
        assert reconstructed.payload == original.payload
        assert reconstructed.parent_ids == original.parent_ids
        assert reconstructed.timestamp == original.timestamp
