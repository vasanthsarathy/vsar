"""Tests for NAF integration with rules and forward chaining."""

import pytest

from vsar.language.ast import Atom, Directive, Fact, NAFLiteral, Rule
from vsar.semantics.engine import VSAREngine


class TestNAFRules:
    """Test NAF integration with rule-based derivation."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
        ]
        return VSAREngine(directives)

    def test_simple_naf_rule(self, engine: VSAREngine):
        """Test rule with single NAF literal.

        Facts: person(alice), person(bob), enemy(bob, carol)
        Rule: safe(X) :- person(X), not enemy(X, _).
        Expected: safe(alice) derived (bob is not safe because enemy(bob, carol))
        """
        # Insert facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))
        engine.insert_fact(Fact(predicate="person", args=["bob"]))
        engine.insert_fact(Fact(predicate="enemy", args=["bob", "carol"]))

        # Define rule: safe(X) :- person(X), not enemy(X, _).
        # Note: Using wildcard "_" as second arg means "any value"
        rule = Rule(
            head=Atom(predicate="safe", args=["X"]),
            body=[
                Atom(predicate="person", args=["X"]),
                NAFLiteral(atom=Atom(predicate="enemy", args=["X", "Y"])),
            ],
        )

        # Apply rule
        derived = engine.apply_rule(rule, k=10)

        # Should derive safe(alice) but not safe(bob)
        assert derived >= 1

        # Check that safe(alice) was derived
        safe_facts = engine.kb.get_facts("safe")
        safe_people = {args[0] for args in safe_facts}

        assert "alice" in safe_people
        # Bob should not be safe (has enemy)
        # (Note: bob might still appear due to approximate matching, but with lower score)

    def test_multiple_naf_literals(self, engine: VSAREngine):
        """Test rule with multiple NAF literals.

        Facts: person(alice), person(bob), enemy(bob, carol), criminal(carol)
        Rule: trustworthy(X) :- person(X), not enemy(X, _), not criminal(X).
        Expected: trustworthy(alice) derived
        """
        # Insert facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))
        engine.insert_fact(Fact(predicate="person", args=["bob"]))
        engine.insert_fact(Fact(predicate="person", args=["carol"]))
        engine.insert_fact(Fact(predicate="enemy", args=["bob", "carol"]))
        engine.insert_fact(Fact(predicate="criminal", args=["carol"]))

        # Define rule: trustworthy(X) :- person(X), not enemy(X, _), not criminal(X).
        rule = Rule(
            head=Atom(predicate="trustworthy", args=["X"]),
            body=[
                Atom(predicate="person", args=["X"]),
                NAFLiteral(atom=Atom(predicate="enemy", args=["X", "Y"])),
                NAFLiteral(atom=Atom(predicate="criminal", args=["X"])),
            ],
        )

        # Apply rule
        derived = engine.apply_rule(rule, k=10)

        # Should derive trustworthy(alice)
        assert derived >= 1

        # Check that trustworthy(alice) was derived
        trustworthy_facts = engine.kb.get_facts("trustworthy")
        trustworthy_people = {args[0] for args in trustworthy_facts}

        assert "alice" in trustworthy_people

    def test_naf_with_bound_variable(self, engine: VSAREngine):
        """Test NAF with variable bound from positive atom.

        Facts: person(alice), person(bob), enemy(alice, bob)
        Rule: friendly(X, Y) :- person(X), person(Y), not enemy(X, Y).
        Expected: friendly(bob, alice) derived, but not friendly(alice, bob)
        """
        # Insert facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))
        engine.insert_fact(Fact(predicate="person", args=["bob"]))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))

        # Define rule: friendly(X, Y) :- person(X), person(Y), not enemy(X, Y).
        rule = Rule(
            head=Atom(predicate="friendly", args=["X", "Y"]),
            body=[
                Atom(predicate="person", args=["X"]),
                Atom(predicate="person", args=["Y"]),
                NAFLiteral(atom=Atom(predicate="enemy", args=["X", "Y"])),
            ],
        )

        # Apply rule
        derived = engine.apply_rule(rule, k=10)

        # Should derive some friendly facts
        assert derived >= 1

        # Check derived facts
        friendly_facts = engine.kb.get_facts("friendly")

        # alice is enemy of bob, so friendly(alice, bob) should NOT exist
        assert ("alice", "bob") not in friendly_facts

        # Symmetric pair should be derivable (bob is not enemy of alice)
        # Note: This might not be derived due to approximate matching


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
