"""Tests for VSAR execution engine."""

from pathlib import Path

import pytest
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import QueryResult, VSAREngine


class TestVSAREngine:
    """Test VSAREngine."""

    def test_create_engine_minimal(self) -> None:
        """Test creating engine with minimal config."""
        directives = []
        engine = VSAREngine(directives)

        # Should use defaults
        assert engine.config.get("backend_type", "FHRR") == "FHRR"
        assert engine.config.get("dim", 8192) == 8192
        assert engine.config.get("seed", 42) == 42

    def test_create_engine_with_model(self) -> None:
        """Test creating engine with model directive."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 100})
        ]
        engine = VSAREngine(directives)

        assert engine.config["backend_type"] == "FHRR"
        assert engine.config["dim"] == 512
        assert engine.config["seed"] == 100

    def test_create_engine_with_threshold(self) -> None:
        """Test creating engine with threshold directive."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
            Directive(name="threshold", params={"value": 0.5}),
        ]
        engine = VSAREngine(directives)

        assert engine.threshold == 0.5

    def test_create_engine_with_beam(self) -> None:
        """Test creating engine with beam directive."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
            Directive(name="beam", params={"width": 100}),
        ]
        engine = VSAREngine(directives)

        assert engine.beam_width == 100

    def test_create_engine_map_backend(self) -> None:
        """Test creating engine with MAP backend."""
        directives = [
            Directive(name="model", params={"type": "MAP", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        assert engine.config["backend_type"] == "MAP"

    def test_create_engine_invalid_backend(self) -> None:
        """Test creating engine with invalid backend raises error."""
        directives = [
            Directive(name="model", params={"type": "INVALID", "dim": 512, "seed": 42})
        ]

        with pytest.raises(ValueError, match="Unknown backend type"):
            VSAREngine(directives)

    def test_insert_fact(self) -> None:
        """Test inserting a fact."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        fact = Fact(predicate="parent", args=["alice", "bob"])
        engine.insert_fact(fact)

        # Verify fact was inserted
        stats = engine.stats()
        assert stats["total_facts"] == 1
        assert "parent" in stats["predicates"]

    def test_insert_multiple_facts(self) -> None:
        """Test inserting multiple facts."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        facts = [
            Fact(predicate="parent", args=["alice", "bob"]),
            Fact(predicate="parent", args=["alice", "carol"]),
            Fact(predicate="parent", args=["bob", "dave"]),
        ]

        for fact in facts:
            engine.insert_fact(fact)

        stats = engine.stats()
        assert stats["total_facts"] == 3
        assert stats["predicates"]["parent"] == 3

    def test_query_simple(self) -> None:
        """Test simple query execution."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        # Insert facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["alice", "carol"]))

        # Query: parent(alice, X)
        query = Query(predicate="parent", args=["alice", None])
        result = engine.query(query, k=5)

        assert isinstance(result, QueryResult)
        assert result.query == query
        assert len(result.results) > 0
        assert isinstance(result.trace_id, str)

    def test_query_results_format(self) -> None:
        """Test query results format."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

        query = Query(predicate="parent", args=["alice", None])
        result = engine.query(query, k=5)

        # Each result should be (entity, score)
        for entity, score in result.results:
            assert isinstance(entity, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_query_with_trace(self) -> None:
        """Test query creates trace events."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

        query = Query(predicate="parent", args=["alice", None])
        result = engine.query(query)

        # Verify trace was created
        trace = engine.trace.get_dag()
        assert len(trace) == 2  # query + retrieval events

        # Verify query event
        query_event = trace[0]
        assert query_event.type == "query"
        assert query_event.payload["predicate"] == "parent"

        # Verify retrieval event
        retrieval_event = trace[1]
        assert retrieval_event.type == "retrieval"
        assert query_event.id in retrieval_event.parent_ids

    def test_query_multiple_variables_raises_error(self) -> None:
        """Test query with multiple variables raises error."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        # Query with 2 variables: parent(X, Y)
        query = Query(predicate="parent", args=[None, None])

        with pytest.raises(ValueError, match="exactly 1 variable"):
            engine.query(query)

    def test_query_no_variables_raises_error(self) -> None:
        """Test query with no variables raises error."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        # Query with no variables: parent(alice, bob)
        query = Query(predicate="parent", args=["alice", "bob"])

        with pytest.raises(ValueError, match="exactly 1 variable"):
            engine.query(query)

    def test_stats(self) -> None:
        """Test getting KB statistics."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        # Initially empty
        stats = engine.stats()
        assert stats["total_facts"] == 0

        # After inserting facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="lives_in", args=["alice", "boston"]))

        stats = engine.stats()
        assert stats["total_facts"] == 2
        assert len(stats["predicates"]) == 2

    def test_export_kb_json(self) -> None:
        """Test exporting KB to JSON."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

        data = engine.export_kb("json")
        assert isinstance(data, dict)

    def test_export_kb_jsonl(self) -> None:
        """Test exporting KB to JSONL."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        data = engine.export_kb("jsonl")
        assert isinstance(data, str)
        lines = data.strip().split("\n")
        assert len(lines) == 2

    def test_export_kb_invalid_format(self) -> None:
        """Test exporting KB with invalid format raises error."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        with pytest.raises(ValueError, match="Invalid export format"):
            engine.export_kb("invalid")

    def test_save_and_load_kb(self, tmp_path: Path) -> None:
        """Test saving and loading KB."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ]
        engine = VSAREngine(directives)

        # Insert facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

        # Save KB
        kb_path = tmp_path / "test.h5"
        engine.save_kb(kb_path)

        assert kb_path.exists()

        # Create new engine and load KB
        engine2 = VSAREngine(directives)
        engine2.load_kb(kb_path)

        # Verify facts were loaded
        stats = engine2.stats()
        assert stats["total_facts"] == 1
