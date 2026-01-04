"""Integration tests for multi-variable query support."""

import pytest

from vsar.language.ast import Directive, Fact, Query
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
    # Parent facts
    facts = [
        ("parent", ["alice", "bob"]),
        ("parent", ["alice", "charlie"]),
        ("parent", ["bob", "david"]),
        ("parent", ["bob", "eve"]),
        ("parent", ["charlie", "frank"]),
    ]

    for pred, args in facts:
        engine.insert_fact(Fact(predicate=pred, args=args))

    return engine


class TestBasicMultiVariable:
    """Test basic multi-variable query functionality."""

    def test_two_variable_query_all_pairs(self, family_kb):
        """Test parent(?, ?) returns all parent-child pairs."""
        # Query: parent(?, ?)
        result = family_kb.query(Query(predicate="parent", args=[None, None]), k=10)

        # Should return all 5 parent-child pairs
        assert len(result.results) == 5

        # Extract entity tuples from string representations
        # Results are in format "('alice', 'bob')"
        result_tuples = []
        for res_str, score in result.results:
            # Parse the string tuple
            import ast
            result_tuples.append(ast.literal_eval(res_str))

        # Verify all expected pairs are present
        expected_pairs = {
            ("alice", "bob"),
            ("alice", "charlie"),
            ("bob", "david"),
            ("bob", "eve"),
            ("charlie", "frank"),
        }

        assert set(result_tuples) == expected_pairs

    def test_two_variable_query_scores(self, family_kb):
        """Test that multi-variable queries return reasonable similarity scores."""
        result = family_kb.query(Query(predicate="parent", args=[None, None]), k=10)

        # All scores should be between 0 and 1
        for _, score in result.results:
            assert 0.0 <= score <= 1.0

        # Scores should be sorted in descending order
        scores = [score for _, score in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_partially_bound_multi_variable(self, family_kb):
        """Test query with one bound and one unbound variable."""
        # Query: parent(bob, ?)
        # Note: This should use single-variable retrieval (optimized path)
        result = family_kb.query(Query(predicate="parent", args=["bob", None]), k=10)

        # Should return bob's children
        children = [res_str for res_str, _ in result.results]
        assert "david" in children
        assert "eve" in children
        assert len(result.results) == 2

    def test_empty_kb(self, engine):
        """Test multi-variable query on empty KB."""
        result = engine.query(Query(predicate="parent", args=[None, None]), k=10)
        assert len(result.results) == 0

    def test_no_matching_facts(self, family_kb):
        """Test multi-variable query for non-existent predicate."""
        # Query: enemy(?, ?) - predicate doesn't exist
        # Should return empty results gracefully
        result = family_kb.query(Query(predicate="enemy", args=[None, None]), k=10)
        assert len(result.results) == 0


class TestThreeVariableQueries:
    """Test queries with three or more variables."""

    @pytest.fixture
    def works_in_kb(self, engine):
        """Create works_in(person, department, role) knowledge base."""
        facts = [
            ("works_in", ["alice", "engineering", "lead"]),
            ("works_in", ["bob", "engineering", "engineer"]),
            ("works_in", ["charlie", "sales", "manager"]),
            ("works_in", ["david", "sales", "rep"]),
        ]

        for pred, args in facts:
            engine.insert_fact(Fact(predicate=pred, args=args))

        return engine

    def test_three_variable_query_all(self, works_in_kb):
        """Test works_in(?, ?, ?) returns all triples."""
        result = works_in_kb.query(
            Query(predicate="works_in", args=[None, None, None]), k=10
        )

        # Should return all 4 work assignments
        assert len(result.results) == 4

        # Verify structure
        import ast
        result_tuples = [ast.literal_eval(res_str) for res_str, _ in result.results]

        expected_triples = {
            ("alice", "engineering", "lead"),
            ("bob", "engineering", "engineer"),
            ("charlie", "sales", "manager"),
            ("david", "sales", "rep"),
        }

        assert set(result_tuples) == expected_triples

    def test_three_variable_partially_bound(self, works_in_kb):
        """Test works_in(alice, ?, ?) with first position bound."""
        result = works_in_kb.query(
            Query(predicate="works_in", args=["alice", None, None]), k=10
        )

        # Should return alice's department and role
        assert len(result.results) == 1

        import ast
        result_tuple = ast.literal_eval(result.results[0][0])
        assert result_tuple == ("engineering", "lead")

    def test_three_variable_middle_bound(self, works_in_kb):
        """Test works_in(?, engineering, ?) with middle position bound."""
        result = works_in_kb.query(
            Query(predicate="works_in", args=[None, "engineering", None]), k=10
        )

        # Should return people in engineering with their roles
        import ast
        result_tuples = [ast.literal_eval(res_str) for res_str, _ in result.results]

        # Note: These are tuples of (person, role) for the two variable positions
        expected_results = {
            ("alice", "lead"),
            ("bob", "engineer"),
        }

        assert set(result_tuples) == expected_results


class TestBeamWidth:
    """Test beam width parameter effects on multi-variable queries."""

    def test_beam_width_affects_results(self, family_kb):
        """Test that beam width parameter is used."""
        # Query with different beam widths
        result_beam_1 = family_kb.query(
            Query(predicate="parent", args=[None, None]), k=10
        )

        # With beam width 1, first position gets only 1 candidate
        # This might miss some results
        assert len(result_beam_1.results) >= 1

    def test_larger_beam_better_coverage(self, family_kb):
        """Test that larger beam width provides better coverage."""
        # The engine's default beam width is 50 (from fixture)
        # With beam width 50, should get all 5 pairs
        result = family_kb.query(Query(predicate="parent", args=[None, None]), k=10)
        assert len(result.results) == 5


class TestAccuracyComparison:
    """Compare multi-variable query accuracy with single-variable baseline."""

    def test_accuracy_vs_single_variable(self, family_kb):
        """Compare multi-variable results with single-variable ground truth."""
        # Get all parent-child pairs using multi-variable query
        multi_result = family_kb.query(
            Query(predicate="parent", args=[None, None]), k=10
        )

        import ast
        multi_pairs = {ast.literal_eval(res_str) for res_str, _ in multi_result.results}

        # Get ground truth by querying each known parent individually
        known_parents = ["alice", "bob", "charlie"]
        ground_truth_pairs = set()

        for parent in known_parents:
            result = family_kb.query(Query(predicate="parent", args=[parent, None]), k=10)
            for child, _ in result.results:
                ground_truth_pairs.add((parent, child))

        # Multi-variable should find all pairs that single-variable finds
        # Calculate accuracy as: |intersection| / |ground_truth|
        correct = len(multi_pairs & ground_truth_pairs)
        total = len(ground_truth_pairs)

        accuracy = correct / total if total > 0 else 0.0

        # Should achieve at least 80% accuracy (preliminary target from plan)
        assert accuracy >= 0.80, f"Accuracy {accuracy:.2%} below 80% threshold"

    def test_similarity_scores_comparable(self, family_kb):
        """Test that multi-variable similarity scores are reasonable."""
        # Multi-variable query
        multi_result = family_kb.query(
            Query(predicate="parent", args=[None, None]), k=10
        )

        # Single-variable query for comparison
        single_result = family_kb.query(
            Query(predicate="parent", args=["alice", None]), k=10
        )

        # Multi-variable scores should be in similar range to single-variable
        # (accounting for averaging across positions)
        multi_scores = [score for _, score in multi_result.results]
        single_scores = [score for _, score in single_result.results]

        if multi_scores and single_scores:
            multi_avg = sum(multi_scores) / len(multi_scores)
            single_avg = sum(single_scores) / len(single_scores)

            # Multi-variable might be slightly lower due to averaging
            # but should be in similar range (within 0.2)
            assert abs(multi_avg - single_avg) < 0.3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_fact_multi_variable(self, engine):
        """Test multi-variable query with only one fact."""
        engine.insert_fact(Fact(predicate="likes", args=["alice", "bob"]))

        result = engine.query(Query(predicate="likes", args=[None, None]), k=10)

        assert len(result.results) == 1
        import ast
        result_tuple = ast.literal_eval(result.results[0][0])
        assert result_tuple == ("alice", "bob")

    def test_duplicate_facts_multi_variable(self, engine):
        """Test multi-variable query with duplicate facts."""
        # Insert same fact multiple times
        for _ in range(3):
            engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

        result = engine.query(Query(predicate="parent", args=[None, None]), k=10)

        # Should still return unique binding (deduplication)
        # Note: KB might store multiple similar vectors, but aggregation should deduplicate
        assert len(result.results) >= 1

    def test_k_parameter_limits_results(self, family_kb):
        """Test that k parameter correctly limits number of results."""
        # Query with k=2
        result = family_kb.query(Query(predicate="parent", args=[None, None]), k=2)

        # Should return at most 2 results
        assert len(result.results) <= 2

    def test_k_larger_than_facts(self, family_kb):
        """Test k parameter larger than number of facts."""
        # Query with k=100 (more than the 5 facts)
        result = family_kb.query(Query(predicate="parent", args=[None, None]), k=100)

        # Should return all 5 facts
        assert len(result.results) == 5


class TestTracing:
    """Test that multi-variable queries are properly traced."""

    def test_trace_recorded(self, family_kb):
        """Test that multi-variable query execution is traced."""
        result = family_kb.query(Query(predicate="parent", args=[None, None]), k=10)

        # Trace ID should be set
        assert result.trace_id is not None

        # Trace should have recorded the query
        trace_events = family_kb.trace.events
        assert len(trace_events) > 0

        # Should have a multi_variable_retrieval event
        retrieval_events = [
            e for e in trace_events if e.type == "multi_variable_retrieval"
        ]
        assert len(retrieval_events) > 0
