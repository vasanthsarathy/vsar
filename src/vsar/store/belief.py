"""Paraconsistent belief tracking for VSAR."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Literal:
    """
    A literal: an atom or its negation.

    Attributes:
        predicate: Predicate name
        args: Tuple of argument names
        negated: True if this is ¬L, False if L
    """
    predicate: str
    args: tuple[str, ...]
    negated: bool = False

    def negate(self) -> 'Literal':
        """Return the negation of this literal."""
        return Literal(self.predicate, self.args, not self.negated)

    def to_key(self) -> str:
        """Convert to a string key for indexing."""
        args_str = ",".join(self.args)
        neg_str = "~" if self.negated else ""
        return f"{neg_str}{self.predicate}({args_str})"

    def __repr__(self) -> str:
        """String representation."""
        args_str = ", ".join(self.args)
        neg_str = "¬" if self.negated else ""
        return f"{neg_str}{self.predicate}({args_str})"

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self.predicate, self.args, self.negated))

    def __eq__(self, other) -> bool:
        """Equality check."""
        if not isinstance(other, Literal):
            return False
        return (self.predicate == other.predicate and
                self.args == other.args and
                self.negated == other.negated)


@dataclass
class BeliefState:
    """
    Paraconsistent belief state for a literal L.

    Tracks independent support for L and ¬L, allowing for:
    - Consistency: supp(L) > 0, supp(¬L) = 0
    - Contradiction: supp(L) > 0, supp(¬L) > 0
    - Unknown: supp(L) = 0, supp(¬L) = 0

    Attributes:
        supp_pos: Support for the literal L (0-1)
        supp_neg: Support for the negation ¬L (0-1)

    Example:
        >>> # Consistent belief
        >>> belief = BeliefState(supp_pos=0.9, supp_neg=0.0)
        >>> belief.is_consistent()
        True
        >>>
        >>> # Contradictory belief
        >>> belief2 = BeliefState(supp_pos=0.7, supp_neg=0.5)
        >>> belief2.is_contradictory()
        True
    """
    supp_pos: float = 0.0
    supp_neg: float = 0.0

    def __post_init__(self):
        """Validate belief state."""
        if not (0.0 <= self.supp_pos <= 1.0):
            raise ValueError(f"supp_pos must be in [0, 1], got {self.supp_pos}")
        if not (0.0 <= self.supp_neg <= 1.0):
            raise ValueError(f"supp_neg must be in [0, 1], got {self.supp_neg}")

    def is_consistent(self) -> bool:
        """Check if belief is consistent (only positive support)."""
        return self.supp_pos > 0 and self.supp_neg == 0

    def is_contradictory(self) -> bool:
        """Check if belief is contradictory (both positive and negative support)."""
        return self.supp_pos > 0 and self.supp_neg > 0

    def is_unknown(self) -> bool:
        """Check if belief is unknown (no support either way)."""
        return self.supp_pos == 0 and self.supp_neg == 0

    def net_support(self) -> float:
        """Calculate net support (pos - neg)."""
        return self.supp_pos - self.supp_neg

    def update_positive(self, support: float) -> 'BeliefState':
        """Add support for the positive literal."""
        new_supp = min(1.0, self.supp_pos + support)
        return BeliefState(supp_pos=new_supp, supp_neg=self.supp_neg)

    def update_negative(self, support: float) -> 'BeliefState':
        """Add support for the negative literal."""
        new_supp = min(1.0, self.supp_neg + support)
        return BeliefState(supp_pos=self.supp_pos, supp_neg=new_supp)

    def __repr__(self) -> str:
        """String representation."""
        status = "UNKNOWN"
        if self.is_consistent():
            status = "CONSISTENT"
        elif self.is_contradictory():
            status = "CONTRADICTORY"

        return f"BeliefState(+{self.supp_pos:.2f}, -{self.supp_neg:.2f}) [{status}]"
