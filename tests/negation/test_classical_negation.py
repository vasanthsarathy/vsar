"""Tests for classical negation support."""

import pytest

from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine


class TestClassicalNegation:
    """Test classical negation (strong negation ~)."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
        ]
        return VSAREngine(directives)

    def test_insert_positive_fact(self, engine: VSAREngine):
        """Test inserting a positive fact."""
        fact = Fact(predicate="person", args=["alice"])
        engine.insert_fact(fact)

        # Verify fact was stored
        assert engine.kb.count("person") == 1
        facts = engine.kb.get_facts("person")
        assert ("alice",) in facts

    def test_insert_negative_fact(self, engine: VSAREngine):
        """Test inserting a negative fact: ~enemy(alice, bob)."""
        fact = Fact(predicate="enemy", args=["alice", "bob"], negated=True)
        engine.insert_fact(fact)

        # Negative facts are stored with ~ prefix
        assert engine.kb.count("~enemy") == 1
        facts = engine.kb.get_facts("~enemy")
        assert ("alice", "bob") in facts

    def test_insert_both_positive_and_negative(self, engine: VSAREngine):
        """Test inserting both positive and negative facts for same predicate."""
        # Insert positive fact
        fact_pos = Fact(predicate="enemy", args=["alice", "charlie"])
        engine.insert_fact(fact_pos)

        # Insert negative fact
        fact_neg = Fact(predicate="enemy", args=["alice", "bob"], negated=True)
        engine.insert_fact(fact_neg)

        # Both should be stored separately
        assert engine.kb.count("enemy") == 1  # Positive
        assert engine.kb.count("~enemy") == 1  # Negative

        pos_facts = engine.kb.get_facts("enemy")
        assert ("alice", "charlie") in pos_facts

        neg_facts = engine.kb.get_facts("~enemy")
        assert ("alice", "bob") in neg_facts

    def test_contradiction_same_fact(self, engine: VSAREngine):
        """Test inserting contradictory facts: p(a) and ~p(a)."""
        # Insert positive
        fact_pos = Fact(predicate="enemy", args=["alice", "bob"])
        engine.insert_fact(fact_pos)

        # Insert negative (contradiction!)
        fact_neg = Fact(predicate="enemy", args=["alice", "bob"], negated=True)
        engine.insert_fact(fact_neg)

        # Both should be stored (paraconsistent mode)
        assert engine.kb.count("enemy") == 1
        assert engine.kb.count("~enemy") == 1

    def test_query_positive_fact(self, engine: VSAREngine):
        """Test querying positive facts."""
        # Insert facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["alice", "carol"]))

        # Query
        query = Query(predicate="parent", args=["alice", None])
        result = engine.query(query, k=10)

        # Should find both children (filter out low-score noise)
        high_score_results = [(e, s) for e, s in result.results if s > 0.2]
        assert len(high_score_results) >= 2
        entities = {r[0] for r in high_score_results}
        assert "bob" in entities
        assert "carol" in entities

    def test_query_negative_fact(self, engine: VSAREngine):
        """Test querying negative facts: ~enemy(alice, ?)."""
        # Insert negative facts
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"], negated=True))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "carol"], negated=True))

        # Query for negative facts (use ~ predicate)
        query = Query(predicate="~enemy", args=["alice", None])
        result = engine.query(query, k=10)

        # Should find both (filter out low-score noise)
        high_score_results = [(e, s) for e, s in result.results if s > 0.2]
        assert len(high_score_results) >= 2
        entities = {r[0] for r in high_score_results}
        assert "bob" in entities
        assert "carol" in entities

    def test_negated_query(self, engine: VSAREngine):
        """Test negated query syntax: query ~enemy(alice, ?)?"""
        # Insert negative facts
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"], negated=True))

        # Negated query
        query = Query(predicate="enemy", args=["alice", None], negated=True)

        # Query negated predicate (internally converts to ~predicate)
        internal_query = Query(predicate=f"~{query.predicate}", args=query.args)
        result = engine.query(internal_query, k=10)

        # Filter out noise and check we found bob
        high_score_results = [(e, s) for e, s in result.results if s > 0.2]
        assert len(high_score_results) >= 1
        assert high_score_results[0][0] == "bob"

    def test_fact_repr(self):
        """Test string representation of facts."""
        pos_fact = Fact(predicate="parent", args=["alice", "bob"])
        assert repr(pos_fact) == "fact parent(alice, bob)."

        neg_fact = Fact(predicate="enemy", args=["alice", "bob"], negated=True)
        assert repr(neg_fact) == "fact ~enemy(alice, bob)."

    def test_query_repr(self):
        """Test string representation of queries."""
        pos_query = Query(predicate="parent", args=["alice", None])
        assert repr(pos_query) == "query parent(alice, ?)?"

        neg_query = Query(predicate="enemy", args=["alice", None], negated=True)
        assert repr(neg_query) == "query ~enemy(alice, ?)?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
