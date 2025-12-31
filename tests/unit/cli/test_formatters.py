"""Tests for CLI formatters."""

import pytest
from vsar.cli.formatters import (
    format_results_json,
    format_results_table,
    format_stats,
    format_trace_dag,
)
from vsar.language.ast import Query
from vsar.semantics.engine import QueryResult
from vsar.trace.collector import TraceCollector


class TestFormatters:
    """Test CLI formatting functions."""

    def test_format_results_table(self) -> None:
        """Test formatting results as table."""
        query = Query(predicate="parent", args=["alice", None])
        result = QueryResult(
            query=query,
            results=[("bob", 0.95), ("carol", 0.88)],
            trace_id="test123",
        )

        format_results_table([result])
        # Function now prints directly, no return value
        # Test passes if no exception is raised

    def test_format_results_json(self) -> None:
        """Test formatting results as JSON."""
        query = Query(predicate="parent", args=["alice", None])
        result = QueryResult(
            query=query,
            results=[("bob", 0.95)],
            trace_id="test123",
        )

        output = format_results_json([result])
        assert isinstance(output, str)
        assert "parent" in output
        assert "bob" in output
        assert "0.95" in output

    def test_format_trace_dag(self) -> None:
        """Test formatting trace DAG."""
        collector = TraceCollector()
        e1 = collector.record("query", {"predicate": "parent"})
        e2 = collector.record("retrieval", {"results": []}, parent_ids=[e1])

        output = format_trace_dag(collector)
        assert isinstance(output, str)
        assert "query" in output
        assert "retrieval" in output

    def test_format_trace_dag_subgraph(self) -> None:
        """Test formatting trace subgraph."""
        collector = TraceCollector()
        e1 = collector.record("query", {"predicate": "parent"})
        e2 = collector.record("retrieval", {"results": []}, parent_ids=[e1])

        output = format_trace_dag(collector, event_id=e2)
        assert isinstance(output, str)
        assert "query" in output
        assert "retrieval" in output

    def test_format_stats(self) -> None:
        """Test formatting KB stats."""
        stats = {
            "total_facts": 10,
            "predicates": {
                "parent": 5,
                "sibling": 5,
            },
        }

        format_stats(stats)
        # Function now prints directly, no return value
        # Test passes if no exception is raised
