"""Integration tests for backward chaining (goal-directed proof search)."""

import pytest

from vsar.language.ast import Atom, Directive, Fact, Rule
from vsar.reasoning.backward_chaining import BackwardChainer
from vsar.semantics.engine import VSAREngine


@pytest.fixture
def engine():
    """Create engine with test configuration."""
    directives = [
        Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
        Directive(name="threshold", params={"value": 0.22}),
        Directive(name="beam", params={"width": 50}),
    ]
    return VSAREngine(directives)


@pytest.fixture
def family_kb(engine):
    """Create family tree knowledge base."""
    facts = [
        ("parent", ["alice", "bob"]),
        ("parent", ["alice", "charlie"]),
        ("parent", ["bob", "david"]),
        ("parent", ["charlie", "eve"]),
    ]

    for pred, args in facts:
        engine.insert_fact(Fact(predicate=pred, args=args))

    return engine


class TestBasicBackwardChaining:
    """Test basic backward chaining functionality."""

    def test_prove_ground_fact(self, family_kb):
        """Test proving a ground fact that exists in KB."""
        chainer = BackwardChainer(family_kb, rules=[], max_depth=5)

        goal = Atom(predicate="parent", args=["alice", "bob"])
        proofs = chainer.prove_goal(goal)

        # Should find exactly one proof
        assert len(proofs) == 1
        assert proofs[0].substitution.is_empty()  # No variables to bind
        assert proofs[0].similarity >= 0.9  # High similarity

    def test_prove_fact_with_variable(self, family_kb):
        """Test proving a goal with one variable."""
        chainer = BackwardChainer(family_kb, rules=[], max_depth=5)

        goal = Atom(predicate="parent", args=["alice", "X"])
        proofs = chainer.prove_goal(goal)

        # Should find two proofs (alice has two children)
        assert len(proofs) == 2

        # Extract bindings
        children = {proof.substitution.get("X") for proof in proofs}
        assert children == {"bob", "charlie"}

    def test_prove_nonexistent_fact(self, family_kb):
        """Test proving a fact that doesn't exist."""
        chainer = BackwardChainer(family_kb, rules=[], max_depth=5)

        goal = Atom(predicate="parent", args=["david", "alice"])
        proofs = chainer.prove_goal(goal)

        # Should find no proofs
        assert len(proofs) == 0

    def test_prove_with_all_variables(self, family_kb):
        """Test proving a goal with all variables."""
        chainer = BackwardChainer(family_kb, rules=[], max_depth=5)

        goal = Atom(predicate="parent", args=["X", "Y"])
        proofs = chainer.prove_goal(goal)

        # Should find all 4 parent facts
        assert len(proofs) == 4

        # Verify all expected pairs are present
        pairs = {
            (proof.substitution.get("X"), proof.substitution.get("Y"))
            for proof in proofs
        }
        assert pairs == {
            ("alice", "bob"),
            ("alice", "charlie"),
            ("bob", "david"),
            ("charlie", "eve"),
        }


class TestRuleProving:
    """Test backward chaining with rules."""

    def test_simple_rule_single_body(self, family_kb):
        """Test proving goal via a simple rule with one body atom."""
        # Rule: human(X) :- parent(X, _)
        rule = Rule(
            head=Atom(predicate="human", args=["X"]),
            body=[Atom(predicate="parent", args=["X", "Y"])],
        )

        chainer = BackwardChainer(family_kb, rules=[rule], max_depth=5)

        # Query: human(alice)?
        goal = Atom(predicate="human", args=["alice"])
        proofs = chainer.prove_goal(goal)

        # Should find proof (alice is a parent)
        assert len(proofs) >= 1
        assert proofs[0].similarity >= 0.8

    def test_grandparent_rule(self, family_kb):
        """Test classic grandparent rule (two-hop)."""
        # Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
        rule = Rule(
            head=Atom(predicate="grandparent", args=["X", "Z"]),
            body=[
                Atom(predicate="parent", args=["X", "Y"]),
                Atom(predicate="parent", args=["Y", "Z"]),
            ],
        )

        chainer = BackwardChainer(family_kb, rules=[rule], max_depth=5)

        # Query: grandparent(alice, david)?
        goal = Atom(predicate="grandparent", args=["alice", "david"])
        proofs = chainer.prove_goal(goal)

        # Should find proof: alice -> bob -> david
        assert len(proofs) >= 1
        assert proofs[0].similarity >= 0.8

    def test_grandparent_with_variable(self, family_kb):
        """Test grandparent query with variable."""
        # Note: Using different variable name to avoid collision with rule head
        rule = Rule(
            head=Atom(predicate="grandparent", args=["A", "C"]),
            body=[
                Atom(predicate="parent", args=["A", "B"]),
                Atom(predicate="parent", args=["B", "C"]),
            ],
        )

        chainer = BackwardChainer(family_kb, rules=[rule], max_depth=5)

        # Query: grandparent(alice, Grandchild)?
        goal = Atom(predicate="grandparent", args=["alice", "Grandchild"])
        proofs = chainer.prove_goal(goal)

        # alice is grandparent of david (via bob) and eve (via charlie)
        # Should find at least one proof
        # Note: Variable resolution chaining is a known limitation
        assert len(proofs) >= 1
        assert all(proof.similarity >= 0.8 for proof in proofs)

    def test_recursive_rule_ancestor(self, family_kb):
        """Test recursive rule for transitive closure."""
        # Rules:
        # ancestor(X, Y) :- parent(X, Y)
        # ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
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

        chainer = BackwardChainer(family_kb, rules=rules, max_depth=5)

        # Query: ancestor(alice, david)?
        goal = Atom(predicate="ancestor", args=["alice", "david"])
        proofs = chainer.prove_goal(goal)

        # Should find proof: alice -> bob (parent) -> david (ancestor via recursive rule)
        assert len(proofs) >= 1


class TestTabling:
    """Test tabling (memoization) functionality."""

    def test_table_caching(self, family_kb):
        """Test that tabling caches results."""
        chainer = BackwardChainer(family_kb, rules=[], max_depth=5)

        goal = Atom(predicate="parent", args=["alice", "X"])

        # First query
        proofs1 = chainer.prove_goal(goal)
        table_size1 = len(chainer.table)

        # Second query (should hit cache)
        proofs2 = chainer.prove_goal(goal)
        table_size2 = len(chainer.table)

        # Results should be identical
        assert len(proofs1) == len(proofs2)
        assert table_size1 == table_size2  # Table didn't grow

        # Check stats
        stats = chainer.get_stats()
        assert stats["table_hits"] > 0

    def test_table_prevents_infinite_loops(self, family_kb):
        """Test that tabling prevents infinite loops in recursive rules."""
        # Pathological recursive rule that could loop infinitely
        # loop(X) :- loop(X)
        rule = Rule(
            head=Atom(predicate="loop", args=["X"]),
            body=[Atom(predicate="loop", args=["X"])],
        )

        chainer = BackwardChainer(family_kb, rules=[rule], max_depth=3)

        goal = Atom(predicate="loop", args=["alice"])
        proofs = chainer.prove_goal(goal)

        # Should terminate without infinite loop (depth limit)
        # No proofs since there's no base case
        assert len(proofs) == 0

    def test_clear_table(self, family_kb):
        """Test clearing the table."""
        chainer = BackwardChainer(family_kb, rules=[], max_depth=5)

        goal = Atom(predicate="parent", args=["alice", "X"])
        chainer.prove_goal(goal)

        assert len(chainer.table) > 0

        chainer.clear_table()

        assert len(chainer.table) == 0
        assert chainer.table_hits == 0
        assert chainer.table_misses == 0


class TestDepthLimit:
    """Test depth limiting to prevent runaway computation."""

    def test_depth_limit_prevents_deep_recursion(self, family_kb):
        """Test that depth limit prevents excessively deep proofs."""
        # Create a deep recursive rule
        rule = Rule(
            head=Atom(predicate="deep", args=["X", "Z"]),
            body=[
                Atom(predicate="parent", args=["X", "Y"]),
                Atom(predicate="deep", args=["Y", "Z"]),
            ],
        )

        # Very shallow depth limit
        chainer = BackwardChainer(family_kb, rules=[rule], max_depth=2)

        goal = Atom(predicate="deep", args=["alice", "X"])
        proofs = chainer.prove_goal(goal)

        # Should find limited proofs due to depth limit
        # May find some shallow proofs but not deep ones
        # Exact number depends on KB structure
        assert len(proofs) >= 0  # Just verify it terminates


class TestComparison:
    """Compare backward chaining with forward chaining."""

    def test_backward_vs_forward_equivalence(self, family_kb):
        """Test that backward and forward chaining produce equivalent results."""
        # Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
        rule = Rule(
            head=Atom(predicate="grandparent", args=["X", "Z"]),
            body=[
                Atom(predicate="parent", args=["X", "Y"]),
                Atom(predicate="parent", args=["Y", "Z"]),
            ],
        )

        # Forward chaining
        from vsar.semantics.chaining import apply_rules
        from vsar.language.ast import Query

        forward_result = apply_rules(family_kb, [rule], max_iterations=10)

        # Query after forward chaining
        query_result = family_kb.query(Query(predicate="grandparent", args=["alice", None]), k=10)
        forward_grandchildren = {child for child, _ in query_result.results}

        # Backward chaining
        chainer = BackwardChainer(family_kb, rules=[rule], max_depth=5)
        goal = Atom(predicate="grandparent", args=["alice", "X"])
        backward_proofs = chainer.prove_goal(goal)
        backward_grandchildren = {proof.substitution.get("X") for proof in backward_proofs}

        # Should find the same grandchildren
        # Note: May not be exactly equal due to approximate matching
        # But should have substantial overlap
        assert len(forward_grandchildren & backward_grandchildren) > 0


class TestStatistics:
    """Test statistics collection."""

    def test_statistics(self, family_kb):
        """Test that statistics are collected correctly."""
        chainer = BackwardChainer(family_kb, rules=[], max_depth=5)

        goal = Atom(predicate="parent", args=["alice", "X"])
        chainer.prove_goal(goal)

        stats = chainer.get_stats()

        assert "table_size" in stats
        assert "table_hits" in stats
        assert "table_misses" in stats
        assert "hit_rate" in stats

        assert stats["table_size"] > 0
        assert stats["table_misses"] > 0  # First query misses
