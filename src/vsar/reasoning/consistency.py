"""Consistency checking for knowledge bases with negation."""

from dataclasses import dataclass
from typing import Optional

from ..kb.store import KnowledgeBase


@dataclass
class Contradiction:
    """A detected contradiction between p(args) and ~p(args)."""

    predicate: str
    args: tuple
    positive_count: int
    negative_count: int

    def __repr__(self) -> str:
        """String representation."""
        args_str = ", ".join(str(a) for a in self.args)
        return (
            f"Contradiction: {self.predicate}({args_str}) "
            f"[+{self.positive_count}, -{self.negative_count}]"
        )


@dataclass
class ConsistencyReport:
    """Result of consistency check."""

    is_consistent: bool
    contradictions: list[Contradiction]
    num_facts: int
    num_contradictions: int

    def __repr__(self) -> str:
        """String representation."""
        if self.is_consistent:
            return f"✓ KB is consistent ({self.num_facts} facts)"
        else:
            return (
                f"✗ KB has {self.num_contradictions} contradictions "
                f"out of {self.num_facts} facts"
            )

    def summary(self) -> str:
        """Detailed summary of contradictions."""
        if self.is_consistent:
            return "Knowledge base is consistent - no contradictions detected."

        lines = [
            f"Consistency Check Results:",
            f"  Total facts: {self.num_facts}",
            f"  Contradictions: {self.num_contradictions}",
            f"",
            f"Detected Contradictions:",
        ]

        for i, contra in enumerate(self.contradictions, 1):
            args_str = ", ".join(str(a) for a in contra.args)
            lines.append(
                f"  {i}. {contra.predicate}({args_str}): "
                f"positive={contra.positive_count}, negative={contra.negative_count}"
            )

        return "\n".join(lines)


class ConsistencyChecker:
    """
    Check KB consistency and detect contradictions.

    Detects cases where both p(args) and ~p(args) exist in the KB,
    which represents a logical contradiction.

    In paraconsistent mode, contradictions are allowed but reported.
    In strict mode, contradictions would cause errors (not yet implemented).

    Args:
        kb: Knowledge base to check

    Example:
        >>> checker = ConsistencyChecker(engine.kb)
        >>> report = checker.check()
        >>> if not report.is_consistent:
        ...     print(report.summary())
    """

    def __init__(self, kb: KnowledgeBase):
        """Initialize consistency checker.

        Args:
            kb: Knowledge base to check
        """
        self.kb = kb

    def check(self) -> ConsistencyReport:
        """
        Check KB for contradictions.

        Scans all predicates for cases where both p(args) and ~p(args) exist.

        Returns:
            ConsistencyReport with all detected contradictions

        Example:
            >>> report = checker.check()
            >>> report.is_consistent
            False
            >>> len(report.contradictions)
            2
        """
        contradictions = []

        # Get all predicates (both positive and negative)
        all_predicates = self.kb.predicates()

        # Find positive predicates that have corresponding negative predicates
        positive_predicates = [p for p in all_predicates if not p.startswith("~")]

        for pred in positive_predicates:
            neg_pred = f"~{pred}"

            # Skip if no negative version exists
            if neg_pred not in all_predicates:
                continue

            # Get facts from both positive and negative
            pos_facts = set(self.kb.get_facts(pred))
            neg_facts = set(self.kb.get_facts(neg_pred))

            # Find contradictions (same args in both)
            common_facts = pos_facts.intersection(neg_facts)

            for args in common_facts:
                contradictions.append(
                    Contradiction(
                        predicate=pred,
                        args=args,
                        positive_count=1,  # Each fact appears once
                        negative_count=1,
                    )
                )

        total_facts = sum(self.kb.count(p) for p in all_predicates)

        return ConsistencyReport(
            is_consistent=len(contradictions) == 0,
            contradictions=contradictions,
            num_facts=total_facts,
            num_contradictions=len(contradictions),
        )

    def get_contradictions_for_predicate(self, predicate: str) -> list[Contradiction]:
        """
        Get all contradictions for a specific predicate.

        Args:
            predicate: Predicate to check (without ~ prefix)

        Returns:
            List of contradictions for this predicate

        Example:
            >>> contradictions = checker.get_contradictions_for_predicate("enemy")
            >>> len(contradictions)
            1
        """
        neg_pred = f"~{predicate}"

        # Check if both exist
        if predicate not in self.kb.predicates() or neg_pred not in self.kb.predicates():
            return []

        pos_facts = set(self.kb.get_facts(predicate))
        neg_facts = set(self.kb.get_facts(neg_pred))

        common_facts = pos_facts.intersection(neg_facts)

        return [
            Contradiction(
                predicate=predicate,
                args=args,
                positive_count=1,
                negative_count=1,
            )
            for args in common_facts
        ]

    def has_contradiction(self, predicate: str, args: tuple) -> bool:
        """
        Check if a specific fact has a contradiction.

        Args:
            predicate: Predicate name (without ~ prefix)
            args: Fact arguments

        Returns:
            True if both p(args) and ~p(args) exist

        Example:
            >>> checker.has_contradiction("enemy", ("alice", "bob"))
            True
        """
        neg_pred = f"~{predicate}"

        # Check if both predicates exist
        if predicate not in self.kb.predicates() or neg_pred not in self.kb.predicates():
            return False

        pos_facts = self.kb.get_facts(predicate)
        neg_facts = self.kb.get_facts(neg_pred)

        return args in pos_facts and args in neg_facts

    def enforce_consistency(self, strategy: str = "keep_positive") -> int:
        """
        Enforce consistency by removing contradictions.

        WARNING: This modifies the KB by removing facts!

        Strategies:
        - "keep_positive": Remove negative facts, keep positive
        - "keep_negative": Remove positive facts, keep negative
        - "remove_both": Remove both positive and negative (conservative)

        Args:
            strategy: Resolution strategy

        Returns:
            Number of facts removed

        Raises:
            ValueError: If strategy is unknown

        Example:
            >>> removed = checker.enforce_consistency("keep_positive")
            >>> removed
            2  # 2 negative facts removed
        """
        if strategy not in ["keep_positive", "keep_negative", "remove_both"]:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Use 'keep_positive', 'keep_negative', or 'remove_both'"
            )

        # For now, return 0 as we need to implement KB modification
        # This requires adding remove() method to KnowledgeBase
        # TODO: Implement KB modification methods
        raise NotImplementedError(
            "Enforcement requires KB modification methods (not yet implemented)"
        )
