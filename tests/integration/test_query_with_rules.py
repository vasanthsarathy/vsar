"""Integration tests for querying with rules."""

import pytest

from vsar.language.ast import Atom, Directive, Fact, Query, Rule
from vsar.semantics.engine import VSAREngine


class TestQueryWithRules:
    """Test query execution with rule application."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
            Directive(name="novelty", params={"threshold": 0.95}),
        ]
        return VSAREngine(directives)

    def test_query_without_rules_baseline(self, engine: VSAREngine) -> None:
        """Test basic query without rules (baseline)."""
        # Insert base facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

        # Query without rules
        result = engine.query(Query(predicate="parent", args=["alice", None]))

        assert len(result.results) > 0
        assert result.query.predicate == "parent"

    def test_query_with_single_rule(self, engine: VSAREngine) -> None:
        """Test query with a single rule that derives facts."""
        # Insert base facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))
        engine.insert_fact(Fact(predicate="person", args=["bob"]))

        # Define rule: human(X) :- person(X).
        rules = [
            Rule(
                head=Atom(predicate="human", args=["X"]),
                body=[Atom(predicate="person", args=["X"])],
            )
        ]

        # Query with rules - should derive human facts first
        result = engine.query(Query(predicate="human", args=[None]), rules=rules)

        # Should find derived facts
        assert len(result.results) >= 2
        entities = [entity for entity, score in result.results]
        assert "alice" in entities or "bob" in entities

    def test_query_grandparent_with_rules(self, engine: VSAREngine) -> None:
        """Test querying for grandparents derived from parent facts."""
        # Insert parent facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        # Define grandparent rule
        rules = [
            Rule(
                head=Atom(predicate="grandparent", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="parent", args=["Y", "Z"]),
                ],
            )
        ]

        # Query for alice's grandchildren with rules
        result = engine.query(
            Query(predicate="grandparent", args=["alice", None]), rules=rules, k=10
        )

        # Should find carol as grandchild
        assert len(result.results) >= 1
        entities = [entity for entity, score in result.results]
        # Carol should be in results (alice -> bob -> carol)
        assert any("carol" in entity for entity in entities)

    def test_query_transitive_closure_with_rules(self, engine: VSAREngine) -> None:
        """Test querying for ancestors with transitive closure rules."""
        # Insert parent facts (chain)
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))
        engine.insert_fact(Fact(predicate="parent", args=["carol", "dave"]))

        # Define ancestor rules (transitive closure)
        rules = [
            Rule(
                head=Atom(predicate="ancestor", args=["X", "Y"]),
                body=[Atom(predicate="parent", args=["X", "Y"])],
            ),
            Rule(
                head=Atom(predicate="ancestor", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="ancestor", args=["Y", "Z"]),
                ],
            ),
        ]

        # Query for alice's ancestors
        result = engine.query(
            Query(predicate="ancestor", args=["alice", None]), rules=rules, k=10
        )

        # Should find all descendants: bob, carol, dave
        assert len(result.results) >= 3

    def test_query_with_multiple_rules(self, engine: VSAREngine) -> None:
        """Test query with multiple rules."""
        # Insert facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        # Multiple rules
        rules = [
            # Child is reverse of parent
            Rule(
                head=Atom(predicate="child", args=["Y", "X"]),
                body=[Atom(predicate="parent", args=["X", "Y"])],
            ),
            # Grandparent
            Rule(
                head=Atom(predicate="grandparent", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="parent", args=["Y", "Z"]),
                ],
            ),
        ]

        # Query for children
        result = engine.query(Query(predicate="child", args=[None, "alice"]), rules=rules)

        # Should find bob as child of alice
        assert len(result.results) >= 1

    def test_query_with_rules_creates_trace(self, engine: VSAREngine) -> None:
        """Test that querying with rules creates proper trace events."""
        # Insert base facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))

        # Define rule
        rules = [
            Rule(
                head=Atom(predicate="human", args=["X"]),
                body=[Atom(predicate="person", args=["X"])],
            )
        ]

        # Query with rules
        result = engine.query(Query(predicate="human", args=[None]), rules=rules)

        # Verify trace was created
        trace = engine.trace.get_dag()
        assert len(trace) >= 2  # At least query + chaining events

        # Find query event
        query_events = [e for e in trace if e.type == "query"]
        assert len(query_events) >= 1
        query_event = query_events[0]
        assert query_event.payload["has_rules"] is True

        # Find chaining event
        chaining_events = [e for e in trace if e.type == "chaining"]
        assert len(chaining_events) >= 1
        chaining_event = chaining_events[0]
        assert chaining_event.payload["num_rules"] == 1
        assert chaining_event.payload["total_derived"] >= 1

    def test_query_with_no_matching_rules(self, engine: VSAREngine) -> None:
        """Test query with rules that don't match any facts."""
        # Insert facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))

        # Rule that won't match
        rules = [
            Rule(
                head=Atom(predicate="animal", args=["X"]),
                body=[Atom(predicate="dog", args=["X"])],
            )
        ]

        # Query with non-matching rules
        # Rules don't derive any facts, so predicate "animal" won't exist
        # This should raise an error
        with pytest.raises(ValueError, match="Predicate 'animal' not found in KB"):
            engine.query(Query(predicate="animal", args=[None]), rules=rules)

    def test_query_without_rules_parameter(self, engine: VSAREngine) -> None:
        """Test that query works without rules parameter (backward compatible)."""
        # Insert facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

        # Query without rules parameter (old API)
        result = engine.query(Query(predicate="parent", args=["alice", None]))

        # Should work as before
        assert len(result.results) >= 1

    def test_query_with_empty_rules_list(self, engine: VSAREngine) -> None:
        """Test query with empty rules list."""
        # Insert facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

        # Query with empty rules list
        result = engine.query(Query(predicate="parent", args=["alice", None]), rules=[])

        # Should return base facts (no derivation attempted)
        assert len(result.results) >= 1

    def test_query_combines_base_and_derived_facts(self, engine: VSAREngine) -> None:
        """Test that query returns both base and derived facts."""
        # Insert some parent facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        # Manually add a grandparent fact (base)
        engine.insert_fact(Fact(predicate="grandparent", args=["eve", "frank"]))

        # Define grandparent rule that will derive more
        rules = [
            Rule(
                head=Atom(predicate="grandparent", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="parent", args=["Y", "Z"]),
                ],
            )
        ]

        # Query for grandparents of a specific person (single variable)
        result = engine.query(Query(predicate="grandparent", args=["alice", None]), rules=rules)

        # Should find derived grandchild (carol)
        assert len(result.results) >= 1

        # Query for grandparents with eve as grandparent (base fact)
        result2 = engine.query(Query(predicate="grandparent", args=["eve", None]), rules=rules)

        # Should find frank (base fact)
        assert len(result2.results) >= 1

    def test_query_with_k_parameter_and_rules(self, engine: VSAREngine) -> None:
        """Test that k parameter works correctly with rules."""
        # Insert multiple facts
        for i in range(10):
            engine.insert_fact(Fact(predicate="person", args=[f"person{i}"]))

        # Rule to derive humans
        rules = [
            Rule(
                head=Atom(predicate="human", args=["X"]),
                body=[Atom(predicate="person", args=["X"])],
            )
        ]

        # Query with k=5
        result = engine.query(Query(predicate="human", args=[None]), rules=rules, k=5)

        # Should respect k limit
        assert len(result.results) <= 5

    def test_multiple_queries_with_same_rules(self, engine: VSAREngine) -> None:
        """Test running multiple queries with the same rules."""
        # Insert facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        # Rules
        rules = [
            Rule(
                head=Atom(predicate="grandparent", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="parent", args=["Y", "Z"]),
                ],
            )
        ]

        # First query
        result1 = engine.query(
            Query(predicate="grandparent", args=["alice", None]), rules=rules
        )

        # Second query (rules already applied, facts already derived)
        result2 = engine.query(
            Query(predicate="grandparent", args=["alice", None]), rules=rules
        )

        # Both should return same results
        # (Note: novelty detection prevents duplicate derivations)
        assert len(result1.results) >= 1
        assert len(result2.results) >= 1
