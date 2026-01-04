"""Tests for negation-as-failure (NAF)."""

import pytest

from vsar.language.ast import Atom, Directive, Fact, NAFLiteral
from vsar.semantics.engine import VSAREngine
from vsar.semantics.substitution import Substitution
from vsar.reasoning.naf import (
    evaluate_naf,
    has_naf_literals,
    separate_positive_and_naf,
)


class TestNAFEvaluation:
    """Test NAF evaluation with threshold-based semantics."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
        ]
        return VSAREngine(directives)

    def test_naf_succeeds_when_fact_absent(self, engine: VSAREngine):
        """Test NAF succeeds when fact is not in KB."""
        # Insert: person(alice)
        engine.insert_fact(Fact(predicate="person", args=["alice"]))

        # NAF literal: not enemy(alice, bob)
        naf_lit = NAFLiteral(atom=Atom(predicate="enemy", args=["alice", "bob"]))

        # Bindings: (empty, atom is already ground)
        bindings = Substitution()

        # Evaluate NAF - should succeed (enemy not in KB)
        result = evaluate_naf(naf_lit, bindings, engine, threshold=0.5)
        assert result is True

    def test_naf_fails_when_fact_present(self, engine: VSAREngine):
        """Test NAF fails when fact is in KB with strong score."""
        # Insert: enemy(alice, bob)
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))

        # NAF literal: not enemy(alice, bob)
        naf_lit = NAFLiteral(atom=Atom(predicate="enemy", args=["alice", "bob"]))

        # Bindings: (empty)
        bindings = Substitution()

        # Evaluate NAF - should fail (enemy IS in KB)
        result = evaluate_naf(naf_lit, bindings, engine, threshold=0.5)
        assert result is False

    def test_naf_with_variable_binding(self, engine: VSAREngine):
        """Test NAF with variable that gets bound."""
        # Insert: enemy(alice, bob)
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))

        # NAF literal: not enemy(X, bob)
        naf_lit = NAFLiteral(atom=Atom(predicate="enemy", args=["X", "bob"]))

        # Bindings: {X: "alice"}
        bindings = Substitution().bind("X", "alice")

        # After binding: not enemy(alice, bob)
        # Should fail (alice IS enemy of bob)
        result = evaluate_naf(naf_lit, bindings, engine, threshold=0.5)
        assert result is False

    def test_naf_with_different_binding(self, engine: VSAREngine):
        """Test NAF succeeds with different variable binding."""
        # Insert: enemy(alice, bob)
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))

        # NAF literal: not enemy(X, carol)
        naf_lit = NAFLiteral(atom=Atom(predicate="enemy", args=["X", "carol"]))

        # Bindings: {X: "alice"}
        bindings = Substitution().bind("X", "alice")

        # After binding: not enemy(alice, carol)
        # Should succeed (alice is NOT enemy of carol)
        result = evaluate_naf(naf_lit, bindings, engine, threshold=0.5)
        assert result is True

    def test_naf_threshold_filters_weak_matches(self, engine: VSAREngine):
        """Test that weak spurious matches (< threshold) don't fail NAF."""
        # Insert: enemy(alice, bob)
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))

        # NAF literal: not enemy(alice, charlie)
        # Query will return weak spurious match for charlie (< 0.5)
        naf_lit = NAFLiteral(atom=Atom(predicate="enemy", args=["alice", "charlie"]))

        bindings = Substitution()

        # With threshold=0.5, weak matches should be ignored
        # NAF should succeed (charlie is "effectively absent")
        result = evaluate_naf(naf_lit, bindings, engine, threshold=0.5)
        assert result is True

    def test_naf_unbound_variable_succeeds_when_no_facts(self, engine: VSAREngine):
        """Test that NAF with unbound variables works correctly.

        NAF now supports wildcards/unbound variables.
        not enemy(X, bob) means "there is no X such that enemy(X, bob)".
        """
        # NAF literal: not enemy(X, bob) with X unbound
        naf_lit = NAFLiteral(atom=Atom(predicate="enemy", args=["X", "bob"]))

        # Empty bindings - X is unbound, acts as wildcard
        bindings = Substitution()

        # Should succeed (no enemy facts at all)
        result = evaluate_naf(naf_lit, bindings, engine, threshold=0.5)
        assert result is True

    def test_naf_unbound_variable_fails_when_fact_exists(self, engine: VSAREngine):
        """Test NAF with unbound variable fails when matching fact exists."""
        # Insert: enemy(alice, bob)
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))

        # NAF literal: not enemy(X, bob) with X unbound
        naf_lit = NAFLiteral(atom=Atom(predicate="enemy", args=["X", "bob"]))

        # Empty bindings - X is unbound, acts as wildcard
        bindings = Substitution()

        # Should fail (enemy(alice, bob) matches pattern enemy(X, bob))
        result = evaluate_naf(naf_lit, bindings, engine, threshold=0.5)
        assert result is False

    def test_naf_repr(self):
        """Test NAF literal string representation."""
        naf_lit = NAFLiteral(atom=Atom(predicate="enemy", args=["X", "bob"]))
        assert repr(naf_lit) == "not enemy(X, bob)"

    def test_naf_get_variables(self):
        """Test getting variables from NAF literal."""
        naf_lit = NAFLiteral(atom=Atom(predicate="enemy", args=["X", "Y"]))
        vars = naf_lit.get_variables()
        assert set(vars) == {"X", "Y"}


class TestNAFHelpers:
    """Test NAF helper functions."""

    def test_has_naf_literals_with_naf(self):
        """Test detecting NAF literals in body."""
        body = [
            Atom(predicate="person", args=["X"]),
            NAFLiteral(atom=Atom(predicate="enemy", args=["X", "_"])),
        ]
        assert has_naf_literals(body) is True

    def test_has_naf_literals_without_naf(self):
        """Test no NAF literals in body."""
        body = [
            Atom(predicate="person", args=["X"]),
            Atom(predicate="lives", args=["X", "city"]),
        ]
        assert has_naf_literals(body) is False

    def test_has_naf_literals_empty_body(self):
        """Test empty body has no NAF."""
        assert has_naf_literals([]) is False

    def test_separate_positive_and_naf(self):
        """Test separating positive atoms from NAF literals."""
        body = [
            Atom(predicate="person", args=["X"]),
            NAFLiteral(atom=Atom(predicate="enemy", args=["X", "_"])),
            Atom(predicate="lives", args=["X", "city"]),
            NAFLiteral(atom=Atom(predicate="banned", args=["X"])),
        ]

        positive, naf = separate_positive_and_naf(body)

        assert len(positive) == 2
        assert len(naf) == 2
        assert all(isinstance(a, Atom) for a in positive)
        assert all(isinstance(n, NAFLiteral) for n in naf)

    def test_separate_all_positive(self):
        """Test separation with only positive atoms."""
        body = [
            Atom(predicate="person", args=["X"]),
            Atom(predicate="lives", args=["X", "city"]),
        ]

        positive, naf = separate_positive_and_naf(body)

        assert len(positive) == 2
        assert len(naf) == 0

    def test_separate_all_naf(self):
        """Test separation with only NAF literals."""
        body = [
            NAFLiteral(atom=Atom(predicate="enemy", args=["X", "_"])),
            NAFLiteral(atom=Atom(predicate="banned", args=["X"])),
        ]

        positive, naf = separate_positive_and_naf(body)

        assert len(positive) == 0
        assert len(naf) == 2


class TestNAFIntegration:
    """Integration tests for NAF."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
        ]
        return VSAREngine(directives)

    def test_safe_person_example(self, engine: VSAREngine):
        """
        Test classic safe person example:
        Facts: person(alice), person(bob), enemy(alice, bob)
        NAF: not enemy(alice, carol) → succeeds
        NAF: not enemy(alice, bob) → fails
        """
        # Insert facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))
        engine.insert_fact(Fact(predicate="person", args=["bob"]))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))

        # Test: not enemy(alice, carol) → should succeed
        naf1 = NAFLiteral(atom=Atom(predicate="enemy", args=["alice", "carol"]))
        result1 = evaluate_naf(naf1, Substitution(), engine, threshold=0.5)
        assert result1 is True  # Alice is NOT enemy of carol

        # Test: not enemy(alice, bob) → should fail
        naf2 = NAFLiteral(atom=Atom(predicate="enemy", args=["alice", "bob"]))
        result2 = evaluate_naf(naf2, Substitution(), engine, threshold=0.5)
        assert result2 is False  # Alice IS enemy of bob


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
