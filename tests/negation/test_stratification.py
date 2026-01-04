"""Tests for stratification analysis."""

import warnings

import pytest

from vsar.language.ast import Atom, Directive, Fact, NAFLiteral, Rule
from vsar.reasoning.stratification import check_stratification, DependencyGraph
from vsar.semantics.chaining import apply_rules
from vsar.semantics.engine import VSAREngine


class TestStratification:
    """Test stratification analysis."""

    def test_stratified_simple_naf(self):
        """Test simple stratified program with NAF.

        Rule: safe(X) :- person(X), not enemy(X, _).
        This is stratified: person and enemy are in stratum 0, safe is in stratum 1.
        """
        rules = [
            Rule(
                head=Atom(predicate="safe", args=["X"]),
                body=[
                    Atom(predicate="person", args=["X"]),
                    NAFLiteral(atom=Atom(predicate="enemy", args=["X", "Y"])),
                ],
            )
        ]

        result = check_stratification(rules)

        assert result.is_stratified
        assert len(result.cycles) == 0
        assert "safe" in result.strata
        assert result.strata["safe"] > result.strata.get("enemy", 0)

    def test_stratified_multiple_strata(self):
        """Test program with multiple strata.

        Rules:
          1. grandparent(X, Z) :- parent(X, Y), parent(Y, Z).  [Stratum 0]
          2. safe(X) :- person(X), not enemy(X, _).             [Stratum 1]
          3. verified(X) :- safe(X), not safe(Y), different(X, Y). [Stratum 2]

        Stratum assignment:
          - Stratum 0: parent, person, enemy, criminal, different (base predicates)
          - Stratum 1: grandparent, safe (depends negatively on stratum 0)
          - Stratum 2: verified (depends negatively on safe from stratum 1)
        """
        rules = [
            Rule(
                head=Atom(predicate="grandparent", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="parent", args=["Y", "Z"]),
                ],
            ),
            Rule(
                head=Atom(predicate="safe", args=["X"]),
                body=[
                    Atom(predicate="person", args=["X"]),
                    NAFLiteral(atom=Atom(predicate="enemy", args=["X", "Y"])),
                ],
            ),
            Rule(
                head=Atom(predicate="verified", args=["X"]),
                body=[
                    Atom(predicate="safe", args=["X"]),
                    NAFLiteral(atom=Atom(predicate="safe", args=["Y"])),
                    Atom(predicate="different", args=["X", "Y"]),
                ],
            ),
        ]

        result = check_stratification(rules)

        assert result.is_stratified
        assert len(result.cycles) == 0

        # Check stratum ordering
        assert result.strata["safe"] > result.strata.get("enemy", 0)
        assert result.strata["verified"] > result.strata["safe"]

    def test_non_stratified_simple_cycle(self):
        """Test non-stratified program with simple negative cycle.

        Rules:
          p(X) :- not q(X).
          q(X) :- not p(X).

        This is NOT stratified (p and q depend negatively on each other).
        """
        rules = [
            Rule(
                head=Atom(predicate="p", args=["X"]),
                body=[NAFLiteral(atom=Atom(predicate="q", args=["X"]))],
            ),
            Rule(
                head=Atom(predicate="q", args=["X"]),
                body=[NAFLiteral(atom=Atom(predicate="p", args=["X"]))],
            ),
        ]

        result = check_stratification(rules)

        assert not result.is_stratified
        assert len(result.cycles) > 0
        assert len(result.strata) == 0  # No valid stratification

    def test_non_stratified_indirect_cycle(self):
        """Test non-stratified program with indirect negative cycle.

        Rules:
          p(X) :- not q(X).
          q(X) :- r(X).
          r(X) :- not p(X).

        Cycle: p -/→ q → r -/→ p
        """
        rules = [
            Rule(
                head=Atom(predicate="p", args=["X"]),
                body=[NAFLiteral(atom=Atom(predicate="q", args=["X"]))],
            ),
            Rule(
                head=Atom(predicate="q", args=["X"]),
                body=[Atom(predicate="r", args=["X"])],
            ),
            Rule(
                head=Atom(predicate="r", args=["X"]),
                body=[NAFLiteral(atom=Atom(predicate="p", args=["X"]))],
            ),
        ]

        result = check_stratification(rules)

        assert not result.is_stratified
        assert len(result.cycles) > 0

    def test_positive_cycle_is_ok(self):
        """Test that positive cycles are allowed (not a stratification issue).

        Rules:
          ancestor(X, Y) :- parent(X, Y).
          ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

        This has a positive cycle (ancestor depends on itself positively),
        which is fine for stratification.
        """
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

        result = check_stratification(rules)

        assert result.is_stratified
        assert len(result.cycles) == 0

    def test_stratification_summary(self):
        """Test stratification result summary."""
        rules = [
            Rule(
                head=Atom(predicate="safe", args=["X"]),
                body=[
                    Atom(predicate="person", args=["X"]),
                    NAFLiteral(atom=Atom(predicate="enemy", args=["X", "Y"])),
                ],
            )
        ]

        result = check_stratification(rules)
        summary = result.summary()

        assert "✓" in summary or "stratified" in summary.lower()
        assert "safe" in summary

    def test_non_stratified_summary(self):
        """Test non-stratified result summary."""
        rules = [
            Rule(
                head=Atom(predicate="p", args=["X"]),
                body=[NAFLiteral(atom=Atom(predicate="q", args=["X"]))],
            ),
            Rule(
                head=Atom(predicate="q", args=["X"]),
                body=[NAFLiteral(atom=Atom(predicate="p", args=["X"]))],
            ),
        ]

        result = check_stratification(rules)
        summary = result.summary()

        assert "✗" in summary or "NOT stratified" in summary
        assert "cycle" in summary.lower()

    def test_empty_rules(self):
        """Test stratification of empty rule set."""
        result = check_stratification([])

        assert result.is_stratified
        assert len(result.cycles) == 0
        assert len(result.strata) == 0

    def test_rules_without_naf(self):
        """Test that rules without NAF are always stratified."""
        rules = [
            Rule(
                head=Atom(predicate="grandparent", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="parent", args=["Y", "Z"]),
                ],
            ),
            Rule(
                head=Atom(predicate="sibling", args=["X", "Y"]),
                body=[
                    Atom(predicate="parent", args=["Z", "X"]),
                    Atom(predicate="parent", args=["Z", "Y"]),
                ],
            ),
        ]

        result = check_stratification(rules)

        assert result.is_stratified
        assert len(result.cycles) == 0

    def test_warning_for_non_stratified_program(self):
        """Test that forward chaining warns about non-stratified programs."""
        # Create engine
        engine = VSAREngine([Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})])

        # Non-stratified rules
        rules = [
            Rule(
                head=Atom(predicate="p", args=["X"]),
                body=[NAFLiteral(atom=Atom(predicate="q", args=["X"]))],
            ),
            Rule(
                head=Atom(predicate="q", args=["X"]),
                body=[NAFLiteral(atom=Atom(predicate="p", args=["X"]))],
            ),
        ]

        # Should warn about non-stratified program
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_rules(engine, rules, max_iterations=10)

            assert len(w) == 1
            assert "Non-stratified" in str(w[0].message)


class TestDependencyGraph:
    """Test dependency graph construction."""

    def test_add_rule_positive_dependency(self):
        """Test adding rule with positive dependencies."""
        graph = DependencyGraph()

        rule = Rule(
            head=Atom(predicate="grandparent", args=["X", "Z"]),
            body=[
                Atom(predicate="parent", args=["X", "Y"]),
                Atom(predicate="parent", args=["Y", "Z"]),
            ],
        )

        graph.add_rule(rule)

        assert "grandparent" in graph.predicates
        assert "parent" in graph.predicates
        assert len(graph.dependencies) == 2
        assert all(not dep.negative for dep in graph.dependencies)

    def test_add_rule_negative_dependency(self):
        """Test adding rule with negative dependencies."""
        graph = DependencyGraph()

        rule = Rule(
            head=Atom(predicate="safe", args=["X"]),
            body=[
                Atom(predicate="person", args=["X"]),
                NAFLiteral(atom=Atom(predicate="enemy", args=["X", "Y"])),
            ],
        )

        graph.add_rule(rule)

        assert "safe" in graph.predicates
        assert "person" in graph.predicates
        assert "enemy" in graph.predicates

        # Check for negative dependency
        negative_deps = [dep for dep in graph.dependencies if dep.negative]
        assert len(negative_deps) == 1
        assert negative_deps[0].source == "safe"
        assert negative_deps[0].target == "enemy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
