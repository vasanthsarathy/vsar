"""Integration tests for forward chaining."""

import pytest

from vsar.language.ast import Atom, Directive, Fact, Rule
from vsar.semantics.chaining import ChainingResult, apply_rules
from vsar.semantics.engine import VSAREngine


class TestForwardChaining:
    """Test forward chaining execution."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
            Directive(name="novelty", params={"threshold": 0.95}),
        ]
        return VSAREngine(directives)

    def test_simple_single_rule_chaining(self, engine: VSAREngine) -> None:
        """Test chaining with single rule that derives facts in one iteration."""
        # Insert base facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))
        engine.insert_fact(Fact(predicate="person", args=["bob"]))

        # rule human(X) :- person(X).
        rule = Rule(
            head=Atom(predicate="human", args=["X"]),
            body=[Atom(predicate="person", args=["X"])],
        )

        result = apply_rules(engine, [rule], max_iterations=10, k=10)

        # Should derive 2 human facts in 1 iteration
        assert isinstance(result, ChainingResult)
        assert result.total_derived == 2
        assert result.iterations == 1
        assert result.fixpoint_reached
        assert not result.max_iterations_reached
        assert result.derived_per_iteration == [2]

    def test_grandparent_two_hop_chaining(self, engine: VSAREngine) -> None:
        """Test two-hop chaining with grandparent rule."""
        # Insert parent facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        # rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        rule = Rule(
            head=Atom(predicate="grandparent", args=["X", "Z"]),
            body=[
                Atom(predicate="parent", args=["X", "Y"]),
                Atom(predicate="parent", args=["Y", "Z"]),
            ],
        )

        result = apply_rules(engine, [rule], max_iterations=10, k=10)

        # Should derive grandparent(alice, carol)
        assert result.total_derived >= 1
        assert result.iterations == 1
        assert result.fixpoint_reached

        # Verify derived fact exists
        stats = engine.stats()
        assert "grandparent" in stats["predicates"]

    def test_transitive_closure_multi_iteration(self, engine: VSAREngine) -> None:
        """Test multi-iteration chaining for transitive closure."""
        # Insert parent facts (chain: alice -> bob -> carol -> dave)
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))
        engine.insert_fact(Fact(predicate="parent", args=["carol", "dave"]))

        # Rules for ancestor (transitive closure):
        # rule ancestor(X, Y) :- parent(X, Y).
        # rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
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

        result = apply_rules(engine, rules, max_iterations=10, k=10)

        # Should derive multiple ancestor facts over multiple iterations
        # Iteration 1: ancestor(alice,bob), ancestor(bob,carol), ancestor(carol,dave)
        # Iteration 2: ancestor(alice,carol), ancestor(bob,dave)
        # Iteration 3: ancestor(alice,dave)
        assert result.total_derived >= 3  # At least the direct ancestors
        assert result.iterations >= 1
        assert result.fixpoint_reached

        # Verify ancestor facts exist
        stats = engine.stats()
        assert "ancestor" in stats["predicates"]
        assert stats["predicates"]["ancestor"] >= 3

    def test_fixpoint_detection(self, engine: VSAREngine) -> None:
        """Test that chaining stops when no new facts are derived."""
        # Insert facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))

        # rule human(X) :- person(X).
        rule = Rule(
            head=Atom(predicate="human", args=["X"]),
            body=[Atom(predicate="person", args=["X"])],
        )

        result = apply_rules(engine, [rule], max_iterations=10, k=10)

        # Should stop after first iteration (fixpoint reached)
        assert result.iterations == 1
        assert result.fixpoint_reached
        assert not result.max_iterations_reached
        assert result.derived_per_iteration == [1]

    def test_max_iterations_limit(self, engine: VSAREngine) -> None:
        """Test that chaining respects max iterations limit."""
        # Create a scenario that could chain indefinitely
        # (though with novelty detection it won't)
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

        # rule ancestor(X, Y) :- parent(X, Y).
        rule = Rule(
            head=Atom(predicate="ancestor", args=["X", "Y"]),
            body=[Atom(predicate="parent", args=["X", "Y"])],
        )

        # Set very low max_iterations
        result = apply_rules(engine, [rule], max_iterations=1, k=10)

        # Should stop after 1 iteration
        assert result.iterations == 1
        # May or may not reach fixpoint depending on derivations
        assert len(result.derived_per_iteration) == 1

    def test_multiple_rules_chaining(self, engine: VSAREngine) -> None:
        """Test chaining with multiple rules applied together."""
        # Insert facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        # Multiple rules:
        # rule child(Y, X) :- parent(X, Y).
        # rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        rules = [
            Rule(
                head=Atom(predicate="child", args=["Y", "X"]),
                body=[Atom(predicate="parent", args=["X", "Y"])],
            ),
            Rule(
                head=Atom(predicate="grandparent", args=["X", "Z"]),
                body=[
                    Atom(predicate="parent", args=["X", "Y"]),
                    Atom(predicate="parent", args=["Y", "Z"]),
                ],
            ),
        ]

        result = apply_rules(engine, rules, max_iterations=10, k=10)

        # Should derive both child and grandparent facts
        assert result.total_derived >= 3  # 2 child + 1 grandparent
        assert result.fixpoint_reached

        stats = engine.stats()
        assert "child" in stats["predicates"]
        assert "grandparent" in stats["predicates"]

    def test_empty_rules_list(self, engine: VSAREngine) -> None:
        """Test chaining with empty rules list."""
        engine.insert_fact(Fact(predicate="person", args=["alice"]))

        result = apply_rules(engine, [], max_iterations=10, k=10)

        # Should complete immediately with no derivations
        # No productive iterations, so iterations = 0 and derived_per_iteration = []
        assert result.total_derived == 0
        assert result.iterations == 0
        assert result.fixpoint_reached
        assert result.derived_per_iteration == []

    def test_no_derivable_facts(self, engine: VSAREngine) -> None:
        """Test chaining when rules don't match any facts."""
        # Insert facts but use rule that doesn't match
        engine.insert_fact(Fact(predicate="person", args=["alice"]))

        # rule animal(X) :- dog(X).
        rule = Rule(
            head=Atom(predicate="animal", args=["X"]),
            body=[Atom(predicate="dog", args=["X"])],
        )

        result = apply_rules(engine, [rule], max_iterations=10, k=10)

        # Should complete immediately with no derivations
        # No productive iterations, so iterations = 0
        assert result.total_derived == 0
        assert result.iterations == 0
        assert result.fixpoint_reached

    def test_chaining_preserves_base_facts(self, engine: VSAREngine) -> None:
        """Test that chaining doesn't modify base facts."""
        # Insert base facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        initial_count = engine.kb.count("parent")

        # rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        rule = Rule(
            head=Atom(predicate="grandparent", args=["X", "Z"]),
            body=[
                Atom(predicate="parent", args=["X", "Y"]),
                Atom(predicate="parent", args=["Y", "Z"]),
            ],
        )

        apply_rules(engine, [rule], max_iterations=10, k=10)

        # Parent facts should remain unchanged
        assert engine.kb.count("parent") == initial_count

    def test_chaining_iteration_tracking(self, engine: VSAREngine) -> None:
        """Test that chaining tracks facts derived per iteration."""
        # Create scenario with multi-iteration derivations
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        # Rules for transitive closure
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

        result = apply_rules(engine, rules, max_iterations=10, k=10)

        # Should have multiple iterations
        assert len(result.derived_per_iteration) >= 1

        # First iteration should derive direct ancestors
        assert result.derived_per_iteration[0] >= 2

        # Sum of per-iteration should equal total
        assert sum(result.derived_per_iteration) == result.total_derived

    def test_large_family_tree_chaining(self, engine: VSAREngine) -> None:
        """Test chaining with larger family tree."""
        # Build a larger family tree
        # Generation 1: alice, bob
        # Generation 2: carol, dave (children of alice,bob)
        # Generation 3: eve, frank (children of carol, dave)
        engine.insert_fact(Fact(predicate="parent", args=["alice", "carol"]))
        engine.insert_fact(Fact(predicate="parent", args=["alice", "dave"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "dave"]))
        engine.insert_fact(Fact(predicate="parent", args=["carol", "eve"]))
        engine.insert_fact(Fact(predicate="parent", args=["dave", "frank"]))

        # rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        rule = Rule(
            head=Atom(predicate="grandparent", args=["X", "Z"]),
            body=[
                Atom(predicate="parent", args=["X", "Y"]),
                Atom(predicate="parent", args=["Y", "Z"]),
            ],
        )

        result = apply_rules(engine, [rule], max_iterations=10, k=10)

        # Should derive multiple grandparent facts
        # alice->carol->eve, alice->dave->frank, bob->carol->eve, bob->dave->frank
        assert result.total_derived >= 4
        assert result.fixpoint_reached

        stats = engine.stats()
        assert stats["predicates"]["grandparent"] >= 4


class TestSemiNaiveEvaluation:
    """Test semi-naive evaluation optimization."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
            Directive(name="novelty", params={"threshold": 0.95}),
        ]
        return VSAREngine(directives)

    def test_semi_naive_produces_same_results_as_naive(self, engine: VSAREngine) -> None:
        """Test that semi-naive produces identical results to naive."""
        # Insert parent facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))
        engine.insert_fact(Fact(predicate="parent", args=["carol", "dave"]))

        # Transitive closure rules
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

        # Run naive evaluation
        result_naive = apply_rules(engine, rules, max_iterations=10, k=10, semi_naive=False)

        # Get ancestor facts from naive
        naive_ancestors = set(engine.kb.get_facts("ancestor"))
        naive_count = engine.kb.count("ancestor")

        # Reset engine
        engine.kb.clear()
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))
        engine.insert_fact(Fact(predicate="parent", args=["carol", "dave"]))

        # Run semi-naive evaluation
        result_semi_naive = apply_rules(engine, rules, max_iterations=10, k=10, semi_naive=True)

        # Get ancestor facts from semi-naive
        semi_naive_ancestors = set(engine.kb.get_facts("ancestor"))
        semi_naive_count = engine.kb.count("ancestor")

        # Should produce same total number of facts
        assert result_semi_naive.total_derived == result_naive.total_derived
        assert semi_naive_count == naive_count

        # Should produce same facts (might be in different order)
        assert semi_naive_ancestors == naive_ancestors

        # Both should reach fixpoint
        assert result_naive.fixpoint_reached
        assert result_semi_naive.fixpoint_reached

    def test_semi_naive_grandparent_example(self, engine: VSAREngine) -> None:
        """Test semi-naive on grandparent example."""
        # Insert parent facts
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

        # Grandparent rule
        rule = Rule(
            head=Atom(predicate="grandparent", args=["X", "Z"]),
            body=[
                Atom(predicate="parent", args=["X", "Y"]),
                Atom(predicate="parent", args=["Y", "Z"]),
            ],
        )

        # Run with semi-naive (should work correctly)
        result = apply_rules(engine, [rule], max_iterations=10, k=10, semi_naive=True)

        assert result.total_derived >= 1
        assert result.fixpoint_reached
        assert "grandparent" in engine.kb.predicates()

    def test_semi_naive_handles_empty_rules(self, engine: VSAREngine) -> None:
        """Test that semi-naive handles empty rules list."""
        engine.insert_fact(Fact(predicate="person", args=["alice"]))

        result = apply_rules(engine, [], max_iterations=10, k=10, semi_naive=True)

        assert result.total_derived == 0
        assert result.iterations == 0
        assert result.fixpoint_reached

    def test_semi_naive_handles_no_derivable_facts(self, engine: VSAREngine) -> None:
        """Test semi-naive when rules don't match."""
        engine.insert_fact(Fact(predicate="person", args=["alice"]))

        # Rule that won't match
        rule = Rule(
            head=Atom(predicate="animal", args=["X"]),
            body=[Atom(predicate="dog", args=["X"])],
        )

        result = apply_rules(engine, [rule], max_iterations=10, k=10, semi_naive=True)

        assert result.total_derived == 0
        assert result.iterations == 0
        assert result.fixpoint_reached

    def test_semi_naive_multi_iteration_transitive(self, engine: VSAREngine) -> None:
        """Test semi-naive on multi-iteration transitive closure."""
        # Long chain: alice -> bob -> carol -> dave -> eve
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))
        engine.insert_fact(Fact(predicate="parent", args=["carol", "dave"]))
        engine.insert_fact(Fact(predicate="parent", args=["dave", "eve"]))

        # Transitive closure rules
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

        # Semi-naive should handle this correctly
        result = apply_rules(engine, rules, max_iterations=10, k=10, semi_naive=True)

        # Should derive multiple ancestors
        assert result.total_derived >= 4  # At least direct parents as ancestors
        assert result.iterations >= 1
        assert result.fixpoint_reached

        # Should have derived alice->eve (4 hops)
        ancestors = engine.kb.get_facts("ancestor")
        assert len(ancestors) >= 4

    def test_naive_vs_semi_naive_parameter(self, engine: VSAREngine) -> None:
        """Test that semi_naive parameter controls behavior."""
        engine.insert_fact(Fact(predicate="person", args=["alice"]))

        rule = Rule(
            head=Atom(predicate="human", args=["X"]),
            body=[Atom(predicate="person", args=["X"])],
        )

        # Both modes should work
        result_naive = apply_rules(engine, [rule], semi_naive=False)
        engine.kb.clear_predicate("human")

        result_semi_naive = apply_rules(engine, [rule], semi_naive=True)

        # Should produce same results
        assert result_naive.total_derived == result_semi_naive.total_derived
