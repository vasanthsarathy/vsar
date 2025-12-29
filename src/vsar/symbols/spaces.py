"""Typed symbol spaces for VSAR."""

from enum import Enum


class SymbolSpace(Enum):
    """
    Typed symbol spaces to prevent collisions.

    VSAR uses separate symbol spaces for different types of symbols to reduce
    collisions in the hypervector space. Each space maintains its own basis
    vectors and cleanup memory.
    """

    ENTITIES = "E"
    """Domain entities (e.g., alice, bob, boston)."""

    RELATIONS = "R"
    """Predicates/relations (e.g., parent, lives_in)."""

    ATTRIBUTES = "A"
    """Attributes and literals (e.g., age, color)."""

    CONTEXTS = "C"
    """Context markers for belief modalities (optional, Phase 3+)."""

    TIME = "T"
    """Temporal symbols (optional, Phase 3+)."""

    STRUCTURAL = "S"
    """Structural operators for Clifford mode (optional)."""

    def __str__(self) -> str:
        """Return the symbol space abbreviation."""
        return self.value

    def __repr__(self) -> str:
        """Return a detailed representation."""
        return f"SymbolSpace.{self.name}"
