"""Phase 1 integration tests - end-to-end VSAR flows."""

from pathlib import Path

import pytest
from vsar.language.ast import Directive, Fact, Query
from vsar.language.loader import load_csv, load_facts, load_jsonl, load_vsar
from vsar.language.parser import parse
from vsar.semantics.engine import VSAREngine


class TestPhase1Integration:
    """Integration tests for Phase 1 functionality."""

    def test_complete_vsar_program_flow(self, tmp_path: Path) -> None:
        """Test complete VSAR program execution flow."""
        # Create a VSAR program file
        program_file = tmp_path / "test.vsar"
        program_file.write_text(
            """
            @model FHRR(dim=512, seed=42);
            @threshold(value=0.22);

            fact parent(alice, bob).
            fact parent(alice, carol).
            fact parent(bob, dave).

            query parent(alice, X)?
        """
        )

        # Load and parse program
        program = load_vsar(program_file)
        assert len(program.directives) == 2
        assert len(program.facts) == 3
        assert len(program.queries) == 1

        # Create engine
        engine = VSAREngine(program.directives)

        # Insert facts
        for fact in program.facts:
            engine.insert_fact(fact)

        # Verify insertion
        stats = engine.stats()
        assert stats["total_facts"] == 3
        assert stats["predicates"]["parent"] == 3

        # Execute query
        query_result = engine.query(program.queries[0], k=5)
        assert len(query_result.results) > 0

        # Verify trace
        trace = engine.trace.get_dag()
        assert len(trace) == 2  # query + retrieval events

    def test_csv_ingestion_and_query(self, tmp_path: Path) -> None:
        """Test ingesting CSV facts and querying."""
        # Create CSV file
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text(
            """parent,alice,bob
parent,alice,carol
parent,bob,dave
lives_in,alice,boston
lives_in,bob,cambridge
"""
        )

        # Load facts
        facts = load_csv(csv_file)
        assert len(facts) == 5

        # Create engine
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]
        engine = VSAREngine(directives)

        # Insert facts
        for fact in facts:
            engine.insert_fact(fact)

        # Query parent relationship
        query = Query(predicate="parent", args=["alice", None])
        result = engine.query(query, k=5)

        assert len(result.results) > 0
        entity_names = [entity for entity, _ in result.results]
        assert "bob" in entity_names or "carol" in entity_names

    def test_jsonl_ingestion_and_query(self, tmp_path: Path) -> None:
        """Test ingesting JSONL facts and querying."""
        # Create JSONL file
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text(
            """{"predicate": "parent", "args": ["alice", "bob"]}
{"predicate": "parent", "args": ["alice", "carol"]}
{"predicate": "parent", "args": ["bob", "dave"]}
"""
        )

        # Load facts
        facts = load_jsonl(jsonl_file)
        assert len(facts) == 3

        # Create engine
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]
        engine = VSAREngine(directives)

        # Insert facts
        for fact in facts:
            engine.insert_fact(fact)

        # Query
        query = Query(predicate="parent", args=[None, "dave"])
        result = engine.query(query, k=5)

        assert len(result.results) > 0
        entity_names = [entity for entity, _ in result.results]
        assert "bob" in entity_names

    def test_kb_persistence(self, tmp_path: Path) -> None:
        """Test saving and loading knowledge base."""
        # Create engine and insert facts
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]
        engine1 = VSAREngine(directives)

        facts = [
            Fact(predicate="parent", args=["alice", "bob"]),
            Fact(predicate="parent", args=["bob", "carol"]),
        ]
        for fact in facts:
            engine1.insert_fact(fact)

        # Save KB
        kb_path = tmp_path / "test.h5"
        engine1.save_kb(kb_path)

        assert kb_path.exists()

        # Create new engine and load KB
        engine2 = VSAREngine(directives)
        engine2.load_kb(kb_path)

        # Verify facts were persisted
        stats = engine2.stats()
        assert stats["total_facts"] == 2
        assert stats["predicates"]["parent"] == 2

        # Query the loaded KB
        query = Query(predicate="parent", args=["alice", None])
        result = engine2.query(query, k=5)

        assert len(result.results) > 0

    def test_kb_export_json(self, tmp_path: Path) -> None:
        """Test exporting KB to JSON."""
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]
        engine = VSAREngine(directives)

        facts = [
            Fact(predicate="parent", args=["alice", "bob"]),
            Fact(predicate="sibling", args=["bob", "carol"]),
        ]
        for fact in facts:
            engine.insert_fact(fact)

        # Export as JSON
        data = engine.export_kb("json")
        assert isinstance(data, dict)
        assert "parent" in data
        assert "sibling" in data
        assert len(data["parent"]) == 1
        assert len(data["sibling"]) == 1

    def test_kb_export_jsonl(self, tmp_path: Path) -> None:
        """Test exporting KB to JSONL."""
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]
        engine = VSAREngine(directives)

        facts = [
            Fact(predicate="parent", args=["alice", "bob"]),
            Fact(predicate="parent", args=["bob", "carol"]),
        ]
        for fact in facts:
            engine.insert_fact(fact)

        # Export as JSONL
        data = engine.export_kb("jsonl")
        assert isinstance(data, str)
        lines = data.strip().split("\n")
        assert len(lines) == 2

    def test_auto_format_detection(self, tmp_path: Path) -> None:
        """Test automatic format detection."""
        # CSV file
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text("parent,alice,bob\n")
        csv_facts = load_facts(csv_file, format="auto")
        assert len(csv_facts) == 1

        # JSONL file
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text('{"predicate": "parent", "args": ["alice", "bob"]}\n')
        jsonl_facts = load_facts(jsonl_file, format="auto")
        assert len(jsonl_facts) == 1

        # VSAR file
        vsar_file = tmp_path / "program.vsar"
        vsar_file.write_text("fact parent(alice, bob).\n")
        vsar_facts = load_facts(vsar_file, format="auto")
        assert len(vsar_facts) == 1

    def test_trace_dag_construction(self, tmp_path: Path) -> None:
        """Test trace DAG is correctly constructed."""
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]
        engine = VSAREngine(directives)

        # Insert facts
        facts = [
            Fact(predicate="parent", args=["alice", "bob"]),
            Fact(predicate="parent", args=["bob", "carol"]),
        ]
        for fact in facts:
            engine.insert_fact(fact)

        # Execute multiple queries
        query1 = Query(predicate="parent", args=["alice", None])
        query2 = Query(predicate="parent", args=[None, "carol"])

        result1 = engine.query(query1)
        result2 = engine.query(query2)

        # Verify trace DAG
        trace = engine.trace.get_dag()
        assert len(trace) == 4  # 2 queries + 2 retrieval events

        # Verify query events
        query_events = [e for e in trace if e.type == "query"]
        assert len(query_events) == 2

        # Verify retrieval events
        retrieval_events = [e for e in trace if e.type == "retrieval"]
        assert len(retrieval_events) == 2

        # Verify parent relationships
        for retrieval_event in retrieval_events:
            assert len(retrieval_event.parent_ids) == 1

    def test_deterministic_results(self, tmp_path: Path) -> None:
        """Test queries produce deterministic results with same seed."""
        # Create two engines with same config
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 100})]

        engine1 = VSAREngine(directives)
        engine2 = VSAREngine(directives)

        # Insert same facts to both
        facts = [
            Fact(predicate="parent", args=["alice", "bob"]),
            Fact(predicate="parent", args=["alice", "carol"]),
        ]

        for fact in facts:
            engine1.insert_fact(fact)
            engine2.insert_fact(fact)

        # Execute same query
        query = Query(predicate="parent", args=["alice", None])
        result1 = engine1.query(query, k=5)
        result2 = engine2.query(query, k=5)

        # Results should be identical
        assert len(result1.results) == len(result2.results)
        for (e1, s1), (e2, s2) in zip(result1.results, result2.results):
            assert e1 == e2
            assert abs(s1 - s2) < 1e-6  # Floating point tolerance

    def test_large_scale_ingestion(self, tmp_path: Path) -> None:
        """Test ingestion of larger fact set."""
        # Create 1000 facts
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 1024, "seed": 42})]
        engine = VSAREngine(directives)

        # Generate facts programmatically
        for i in range(1000):
            fact = Fact(predicate="test", args=[f"entity_{i}", f"value_{i}"])
            engine.insert_fact(fact)

        # Verify all facts inserted
        stats = engine.stats()
        assert stats["total_facts"] == 1000
        assert stats["predicates"]["test"] == 1000

        # Query should still work
        query = Query(predicate="test", args=["entity_0", None])
        result = engine.query(query, k=10)
        assert len(result.results) > 0
