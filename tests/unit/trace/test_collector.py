"""Tests for trace collector."""

import pytest
from vsar.trace.collector import TraceCollector
from vsar.trace.events import TraceEvent


class TestTraceCollector:
    """Test TraceCollector."""

    def test_create_collector(self) -> None:
        """Test creating empty collector."""
        collector = TraceCollector()
        assert len(collector.get_dag()) == 0

    def test_record_single_event(self) -> None:
        """Test recording a single event."""
        collector = TraceCollector()
        event_id = collector.record("query", {"predicate": "parent"})

        assert isinstance(event_id, str)
        dag = collector.get_dag()
        assert len(dag) == 1
        assert dag[0].id == event_id
        assert dag[0].type == "query"
        assert dag[0].payload == {"predicate": "parent"}

    def test_record_multiple_events(self) -> None:
        """Test recording multiple events."""
        collector = TraceCollector()
        id1 = collector.record("query", {"step": 1})
        id2 = collector.record("unbind", {"step": 2})
        id3 = collector.record("retrieval", {"step": 3})

        dag = collector.get_dag()
        assert len(dag) == 3
        assert dag[0].id == id1
        assert dag[1].id == id2
        assert dag[2].id == id3

    def test_record_with_parents(self) -> None:
        """Test recording event with parent IDs."""
        collector = TraceCollector()
        parent_id = collector.record("query", {})
        child_id = collector.record("retrieval", {}, parent_ids=[parent_id])

        child = collector.get_event(child_id)
        assert child is not None
        assert child.parent_ids == [parent_id]

    def test_get_event(self) -> None:
        """Test retrieving event by ID."""
        collector = TraceCollector()
        event_id = collector.record("query", {"key": "value"})

        event = collector.get_event(event_id)
        assert event is not None
        assert event.id == event_id
        assert event.payload["key"] == "value"

    def test_get_event_not_found(self) -> None:
        """Test getting non-existent event returns None."""
        collector = TraceCollector()
        event = collector.get_event("nonexistent")
        assert event is None

    def test_get_subgraph_single_event(self) -> None:
        """Test subgraph with single event (no parents)."""
        collector = TraceCollector()
        event_id = collector.record("query", {})

        subgraph = collector.get_subgraph(event_id)
        assert len(subgraph) == 1
        assert subgraph[0].id == event_id

    def test_get_subgraph_linear_chain(self) -> None:
        """Test subgraph with linear parent chain."""
        collector = TraceCollector()
        e1 = collector.record("query", {"step": 1})
        e2 = collector.record("unbind", {"step": 2}, parent_ids=[e1])
        e3 = collector.record("cleanup", {"step": 3}, parent_ids=[e2])

        # Subgraph of e3 should include e1, e2, e3
        subgraph = collector.get_subgraph(e3)
        assert len(subgraph) == 3
        assert subgraph[0].id == e1
        assert subgraph[1].id == e2
        assert subgraph[2].id == e3

    def test_get_subgraph_dag(self) -> None:
        """Test subgraph with DAG structure (multiple parents)."""
        collector = TraceCollector()
        e1 = collector.record("query", {"step": 1})
        e2 = collector.record("unbind", {"step": 2})
        e3 = collector.record("retrieval", {"step": 3}, parent_ids=[e1, e2])

        # Subgraph of e3 should include all three events
        subgraph = collector.get_subgraph(e3)
        assert len(subgraph) == 3
        event_ids = {e.id for e in subgraph}
        assert event_ids == {e1, e2, e3}

    def test_get_subgraph_partial(self) -> None:
        """Test subgraph doesn't include unrelated events."""
        collector = TraceCollector()
        e1 = collector.record("query", {"related": True})
        e2 = collector.record("retrieval", {"related": True}, parent_ids=[e1])
        e3 = collector.record("query", {"related": False})  # Unrelated

        subgraph = collector.get_subgraph(e2)
        assert len(subgraph) == 2
        event_ids = {e.id for e in subgraph}
        assert event_ids == {e1, e2}
        assert e3 not in event_ids

    def test_to_dict(self) -> None:
        """Test converting collector to dictionary."""
        collector = TraceCollector()
        e1 = collector.record("query", {"key": "value"})
        e2 = collector.record("retrieval", {}, parent_ids=[e1])

        data = collector.to_dict()
        assert "events" in data
        assert len(data["events"]) == 2
        assert data["events"][0]["id"] == e1
        assert data["events"][1]["id"] == e2

    def test_from_dict(self) -> None:
        """Test creating collector from dictionary."""
        data = {
            "events": [
                {
                    "id": "evt1",
                    "type": "query",
                    "payload": {"predicate": "parent"},
                    "parent_ids": [],
                    "timestamp": 123.0,
                },
                {
                    "id": "evt2",
                    "type": "retrieval",
                    "payload": {"results": []},
                    "parent_ids": ["evt1"],
                    "timestamp": 124.0,
                },
            ]
        }

        collector = TraceCollector.from_dict(data)
        dag = collector.get_dag()
        assert len(dag) == 2
        assert dag[0].id == "evt1"
        assert dag[1].id == "evt2"
        assert dag[1].parent_ids == ["evt1"]

    def test_roundtrip_serialization(self) -> None:
        """Test to_dict and from_dict roundtrip."""
        original = TraceCollector()
        e1 = original.record("query", {"a": 1})
        e2 = original.record("unbind", {"b": 2}, parent_ids=[e1])

        data = original.to_dict()
        reconstructed = TraceCollector.from_dict(data)

        assert len(reconstructed.get_dag()) == len(original.get_dag())
        for orig_event, recon_event in zip(original.get_dag(), reconstructed.get_dag()):
            assert recon_event.id == orig_event.id
            assert recon_event.type == orig_event.type
            assert recon_event.payload == orig_event.payload
            assert recon_event.parent_ids == orig_event.parent_ids

    def test_clear(self) -> None:
        """Test clearing all events."""
        collector = TraceCollector()
        collector.record("query", {})
        collector.record("retrieval", {})

        assert len(collector.get_dag()) == 2

        collector.clear()
        assert len(collector.get_dag()) == 0

    def test_clear_empty_collector(self) -> None:
        """Test clearing already empty collector."""
        collector = TraceCollector()
        collector.clear()  # Should not raise
        assert len(collector.get_dag()) == 0
