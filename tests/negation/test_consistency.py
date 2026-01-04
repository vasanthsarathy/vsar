"""Tests for consistency checking."""

import pytest

from vsar.language.ast import Directive, Fact
from vsar.semantics.engine import VSAREngine
from vsar.reasoning.consistency import ConsistencyChecker, ConsistencyReport, Contradiction


class TestConsistencyChecker:
    """Test consistency checking functionality."""

    @pytest.fixture
    def engine(self) -> VSAREngine:
        """Create test engine."""
        directives = [
            Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42}),
        ]
        return VSAREngine(directives)

    def test_consistent_kb(self, engine: VSAREngine):
        """Test KB with no contradictions."""
        # Insert only positive facts
        engine.insert_fact(Fact(predicate="person", args=["alice"]))
        engine.insert_fact(Fact(predicate="person", args=["bob"]))
        engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

        # Check consistency
        checker = ConsistencyChecker(engine.kb)
        report = checker.check()

        assert report.is_consistent
        assert len(report.contradictions) == 0
        assert report.num_contradictions == 0
        assert report.num_facts == 3

    def test_detect_single_contradiction(self, engine: VSAREngine):
        """Test detecting a single contradiction."""
        # Insert contradictory facts
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"], negated=True))

        # Check consistency
        checker = ConsistencyChecker(engine.kb)
        report = checker.check()

        assert not report.is_consistent
        assert len(report.contradictions) == 1
        assert report.num_contradictions == 1

        contra = report.contradictions[0]
        assert contra.predicate == "enemy"
        assert contra.args == ("alice", "bob")
        assert contra.positive_count == 1
        assert contra.negative_count == 1

    def test_detect_multiple_contradictions(self, engine: VSAREngine):
        """Test detecting multiple contradictions."""
        # Insert multiple contradictory pairs
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"], negated=True))

        engine.insert_fact(Fact(predicate="enemy", args=["alice", "carol"]))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "carol"], negated=True))

        # Check consistency
        checker = ConsistencyChecker(engine.kb)
        report = checker.check()

        assert not report.is_consistent
        assert len(report.contradictions) == 2
        assert report.num_contradictions == 2

    def test_partial_overlap(self, engine: VSAREngine):
        """Test KB with some contradictions and some consistent facts."""
        # Consistent facts
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "dave"]))
        engine.insert_fact(Fact(predicate="enemy", args=["bob", "carol"], negated=True))

        # Contradictory fact
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"], negated=True))

        checker = ConsistencyChecker(engine.kb)
        report = checker.check()

        assert not report.is_consistent
        assert len(report.contradictions) == 1
        assert report.num_facts == 4

        # Only alice-bob should be contradictory
        contra = report.contradictions[0]
        assert contra.args == ("alice", "bob")

    def test_get_contradictions_for_predicate(self, engine: VSAREngine):
        """Test getting contradictions for a specific predicate."""
        # Add contradictions for "enemy"
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"], negated=True))

        # Add consistent facts for "friend"
        engine.insert_fact(Fact(predicate="friend", args=["alice", "carol"]))

        checker = ConsistencyChecker(engine.kb)

        # Check enemy predicate
        enemy_contradictions = checker.get_contradictions_for_predicate("enemy")
        assert len(enemy_contradictions) == 1
        assert enemy_contradictions[0].predicate == "enemy"

        # Check friend predicate (no contradictions)
        friend_contradictions = checker.get_contradictions_for_predicate("friend")
        assert len(friend_contradictions) == 0

    def test_has_contradiction(self, engine: VSAREngine):
        """Test checking if specific fact has contradiction."""
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"], negated=True))

        engine.insert_fact(Fact(predicate="enemy", args=["alice", "carol"]))
        # No negation for alice-carol

        checker = ConsistencyChecker(engine.kb)

        # alice-bob has contradiction
        assert checker.has_contradiction("enemy", ("alice", "bob"))

        # alice-carol does not
        assert not checker.has_contradiction("enemy", ("alice", "carol"))

        # Non-existent fact
        assert not checker.has_contradiction("enemy", ("bob", "dave"))

    def test_report_repr(self, engine: VSAREngine):
        """Test report string representation."""
        # Consistent KB
        engine.insert_fact(Fact(predicate="person", args=["alice"]))
        checker = ConsistencyChecker(engine.kb)
        report = checker.check()

        assert "✓" in repr(report)
        assert "consistent" in repr(report)

    def test_report_summary_consistent(self, engine: VSAREngine):
        """Test summary for consistent KB."""
        engine.insert_fact(Fact(predicate="person", args=["alice"]))
        checker = ConsistencyChecker(engine.kb)
        report = checker.check()

        summary = report.summary()
        assert "consistent" in summary.lower()
        assert "no contradictions" in summary.lower()

    def test_report_summary_inconsistent(self, engine: VSAREngine):
        """Test summary for inconsistent KB."""
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"], negated=True))

        checker = ConsistencyChecker(engine.kb)
        report = checker.check()

        summary = report.summary()
        assert "Contradiction" in summary
        assert "enemy" in summary
        assert "alice" in summary
        assert "bob" in summary

    def test_contradiction_repr(self):
        """Test contradiction string representation."""
        contra = Contradiction(
            predicate="enemy",
            args=("alice", "bob"),
            positive_count=1,
            negative_count=1,
        )

        repr_str = repr(contra)
        assert "Contradiction" in repr_str
        assert "enemy" in repr_str
        assert "alice" in repr_str
        assert "bob" in repr_str

    def test_empty_kb(self, engine: VSAREngine):
        """Test consistency check on empty KB."""
        checker = ConsistencyChecker(engine.kb)
        report = checker.check()

        assert report.is_consistent
        assert len(report.contradictions) == 0
        assert report.num_facts == 0

    def test_only_negative_facts(self, engine: VSAREngine):
        """Test KB with only negative facts (no contradictions)."""
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"], negated=True))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "carol"], negated=True))

        checker = ConsistencyChecker(engine.kb)
        report = checker.check()

        assert report.is_consistent
        assert len(report.contradictions) == 0

    def test_enforce_consistency_not_implemented(self, engine: VSAREngine):
        """Test that enforce_consistency raises NotImplementedError."""
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"]))
        engine.insert_fact(Fact(predicate="enemy", args=["alice", "bob"], negated=True))

        checker = ConsistencyChecker(engine.kb)

        with pytest.raises(NotImplementedError):
            checker.enforce_consistency("keep_positive")

    def test_enforce_consistency_invalid_strategy(self, engine: VSAREngine):
        """Test that invalid strategy raises ValueError."""
        checker = ConsistencyChecker(engine.kb)

        with pytest.raises(ValueError, match="Unknown strategy"):
            checker.enforce_consistency("invalid_strategy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
