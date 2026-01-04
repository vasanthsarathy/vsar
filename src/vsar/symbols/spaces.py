"""Typed symbol spaces for VSAR v2.0.

This module defines the typed symbol spaces used in the new FHRR-based encoding.
Each space maintains separate basis vectors to prevent collisions and enable
typed cleanup operations.
"""

from enum import Enum


class SymbolSpace(Enum):
    """
    Typed symbol spaces for the new VSAR encoding architecture.

    The new specification uses 11 distinct symbol spaces to organize symbols
    by their semantic role. This enables:
    - Type-safe cleanup (only search relevant space)
    - Reduced collisions (symbols in different spaces don't interfere)
    - Structure-informed decoding (unbind → typed cleanup)
    """

    # Core domain symbols
    ENTITIES = "E"
    """Domain entities / constants (e.g., alice, bob, boston)."""

    CONCEPTS = "C"
    """Unary predicates / concept names (e.g., Person, Doctor)."""

    ROLES = "R"
    """Binary relations (e.g., hasChild, worksAt)."""

    FUNCTIONS = "F"
    """Function symbols (e.g., mother, father)."""

    PREDICATES = "P"
    """General predicates of any arity (e.g., parent, grandparent)."""

    # Structural role markers
    ARG_ROLES = "ARG"
    """Argument position markers (ARG₁, ARG₂, ARG₃, ...)."""

    STRUCT_ROLES = "STRUCT"
    """Structural role markers (HEAD, BODY, SRC, TGT, LEFT, RIGHT)."""

    # Metadata and type tags
    TAGS = "TAG"
    """Type tags (ATOM, TERM, RULE, LIT, META, AXIOM)."""

    # Logical operators
    OPS = "OP"
    """Logical operators (AND, OR, NOT, EXISTS, FORALL)."""

    EPI_OPS = "EPI"
    """Epistemic operators (KNOW, BELIEF) for multi-agent reasoning."""

    GRAPH_OPS = "GRAPH"
    """Argumentation graph operators (EDGE, SUPPORT, ATTACK)."""

    def __str__(self) -> str:
        """Return the symbol space abbreviation."""
        return self.value

    def __repr__(self) -> str:
        """Return a detailed representation."""
        return f"SymbolSpace.{self.name}"
