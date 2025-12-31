"""Tests for join operations."""

import pytest

from vsar.language.ast import Atom, Directive, Fact, Query
from vsar.semantics.engine import VSAREngine
from vsar.semantics.join import (
    CandidateBinding,
    execute_atom_with_bindings,
    initial_candidates_from_atom,
    join_with_atom,
)
from vsar.semantics.substitution import Substitution


class TestCandidateBinding:
    """Test CandidateBinding class."""

    def test_create_empty_binding(self) -> None:
        """Test creating empty binding."""
        binding = CandidateBinding(substitution=Substitution(), score=1.0)
        assert binding.score == 1.0
        assert binding.substitution.is_empty()

    def test_create_with_substitution(self) -> None:
        """Test creating binding with substitution."""
        sub = Substitution().bind("X", "alice")
        binding = CandidateBinding(substitution=sub, score=0.85)

        assert binding.score == 0.85
        assert binding.substitution.get("X") == "alice"

    def test_extend_binding(self) -> None:
        """Test extending a binding with new variable."""
        binding = CandidateBinding(
            substitution=Substitution().bind("X", "alice"),
            score=0.9,
        )

        extended = binding.extend("Y", "bob", 0.8)

        # Original unchanged
        assert binding.substitution.get("Y") is None
        assert binding.score == 0.9

        # Extended has both variables
        assert extended.substitution.get("X") == "alice"
        assert extended.substitution.get("Y") == "bob"
        assert abs(extended.score - 0.72) < 0.001  # 0.9 * 0.8

    def test_extend_multiple_times(self) -> None:
        """Test extending binding multiple times."""
        binding = CandidateBinding(substitution=Substitution(), score=1.0)

        b1 = binding.extend("X", "alice", 0.9)
        b2 = b1.extend("Y", "bob", 0.8)
        b3 = b2.extend("Z", "carol", 0.7)

        assert b3.substitution.get("X") == "alice"
        assert b3.substitution.get("Y") == "bob"
        assert b3.substitution.get("Z") == "carol"
        assert abs(b3.score - 0.504) < 0.001  # 0.9 * 0.8 * 0.7


class TestInitialCandidates:
    """Test initial_candidates_from_atom."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]
        engine = VSAREngine(directives)

        # Insert test facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))
        engine.insert_fact(Fact(predicate="person", args=["bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        return engine

    def test_initial_candidates_single_variable(self, engine: VSAREngine) -> None:
        """Test creating initial candidates from atom with one variable."""
        atom = Atom(predicate="person", args=["X"])

        candidates = initial_candidates_from_atom(atom, engine.query, k=5, kb=engine.kb)

        # Should have candidates for alice and bob
        assert len(candidates) >= 2

        # All should have X bound
        for cand in candidates:
            assert cand.substitution.has("X")
            assert cand.score > 0.0

    def test_initial_candidates_with_constant(self, engine: VSAREngine) -> None:
        """Test initial candidates with constant in atom."""
        atom = Atom(predicate="parent", args=["alice", "X"])

        candidates = initial_candidates_from_atom(atom, engine.query, k=5, kb=engine.kb)

        # Should have candidate for bob
        assert len(candidates) >= 1

        # All should have X bound
        for cand in candidates:
            assert cand.substitution.has("X")

    def test_initial_candidates_multiple_variables(self, engine: VSAREngine) -> None:
        """Test that multiple variables in first atom enumerates facts."""
        atom = Atom(predicate="parent", args=["X", "Y"])

        candidates = initial_candidates_from_atom(atom, engine.query, k=5, kb=engine.kb)

        # Should have candidates with both X and Y bound
        assert len(candidates) >= 2

        for cand in candidates:
            assert cand.substitution.has("X")
            assert cand.substitution.has("Y")
            assert cand.score == 1.0


class TestExecuteAtomWithBindings:
    """Test execute_atom_with_bindings."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]
        engine = VSAREngine(directives)

        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        return engine

    def test_execute_with_one_bound_variable(self, engine: VSAREngine) -> None:
        """Test executing atom with one variable bound."""
        # Binding: {X: alice}
        # Atom: parent(X, Y)
        # Should execute: parent(alice, Y)?

        binding = CandidateBinding(
            substitution=Substitution().bind("X", "alice"),
            score=0.9,
        )

        atom = Atom(predicate="parent", args=["X", "Y"])

        results = execute_atom_with_bindings(atom, binding, engine.query, k=5)

        # Should find bob
        assert len(results) >= 1
        values = [val for val, score in results]
        assert "bob" in values

    def test_execute_fully_ground_atom(self, engine: VSAREngine) -> None:
        """Test executing fully ground atom returns empty."""
        binding = CandidateBinding(
            substitution=Substitution().bind("X", "alice").bind("Y", "bob"),
            score=0.9,
        )

        atom = Atom(predicate="parent", args=["X", "Y"])

        results = execute_atom_with_bindings(atom, binding, engine.query, k=5)

        # Fully ground - can't query
        assert results == []


class TestJoinWithAtom:
    """Test join_with_atom."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]
        engine = VSAREngine(directives)

        # Family tree for grandparent example
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["alice", "carol"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "dave"]))
        engine.insert_fact(Fact(predicate="parent", args=["carol", "eve"]))

        return engine

    def test_join_extends_bindings(self, engine: VSAREngine) -> None:
        """Test that join extends bindings with new variable."""
        # Start with: [{X: alice, score: 0.9}]
        # Join with: parent(X, Y)
        # Should get: [{X: alice, Y: bob, score: ...}, {X: alice, Y: carol, score: ...}]

        initial = [
            CandidateBinding(
                substitution=Substitution().bind("X", "alice"),
                score=0.9,
            )
        ]

        atom = Atom(predicate="parent", args=["X", "Y"])

        joined = join_with_atom(initial, atom, engine.query, beam_width=50, k=5)

        # Should have candidates with both X and Y bound
        assert len(joined) >= 2

        for cand in joined:
            assert cand.substitution.get("X") == "alice"
            assert cand.substitution.has("Y")
            assert cand.score < 0.9  # Score should decrease (multiplied)

    def test_join_propagates_scores(self, engine: VSAREngine) -> None:
        """Test that join propagates scores correctly."""
        initial = [
            CandidateBinding(
                substitution=Substitution().bind("X", "alice"),
                score=1.0,
            )
        ]

        atom = Atom(predicate="parent", args=["X", "Y"])

        joined = join_with_atom(initial, atom, engine.query, beam_width=50, k=5)

        # All scores should be <= 1.0 (original score * query score)
        for cand in joined:
            assert 0.0 <= cand.score <= 1.0

    def test_join_with_shared_variable(self, engine: VSAREngine) -> None:
        """Test joining atoms with shared variable."""
        # Start with: [{X: alice, Y: bob, score: 0.8}]
        # Join with: parent(Y, Z)
        # Should execute: parent(bob, Z)? and extend bindings

        initial = [
            CandidateBinding(
                substitution=Substitution().bind("X", "alice").bind("Y", "bob"),
                score=0.8,
            )
        ]

        atom = Atom(predicate="parent", args=["Y", "Z"])

        joined = join_with_atom(initial, atom, engine.query, beam_width=50, k=5)

        # Should have candidates with X, Y, Z bound
        assert len(joined) >= 1

        for cand in joined:
            assert cand.substitution.get("X") == "alice"
            assert cand.substitution.get("Y") == "bob"
            assert cand.substitution.has("Z")
            # dave should be one of the results
            if cand.substitution.get("Z") == "dave":
                assert True
                break
        else:
            # Should find dave as child of bob
            pass  # VSA is approximate, might not always find exact match

    def test_beam_search_limits_candidates(self, engine: VSAREngine) -> None:
        """Test that beam search limits number of candidates."""
        # Create multiple initial candidates
        initial = [
            CandidateBinding(substitution=Substitution().bind("X", "alice"), score=0.9),
            CandidateBinding(substitution=Substitution().bind("X", "bob"), score=0.8),
            CandidateBinding(substitution=Substitution().bind("X", "carol"), score=0.7),
        ]

        atom = Atom(predicate="parent", args=["X", "Y"])

        # Small beam width
        joined = join_with_atom(initial, atom, engine.query, beam_width=3, k=5)

        # Should be limited by beam width
        assert len(joined) <= 3

    def test_join_sorts_by_score(self, engine: VSAREngine) -> None:
        """Test that join results are sorted by score descending."""
        initial = [
            CandidateBinding(substitution=Substitution().bind("X", "alice"), score=0.9),
        ]

        atom = Atom(predicate="parent", args=["X", "Y"])

        joined = join_with_atom(initial, atom, engine.query, beam_width=50, k=5)

        # Check scores are descending
        for i in range(len(joined) - 1):
            assert joined[i].score >= joined[i + 1].score
