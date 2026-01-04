"""Substitution management for unification."""

from typing import Optional
from dataclasses import dataclass


@dataclass
class Term:
    """Base class for logical terms."""
    name: str

    def apply_substitution(self, subst: 'Substitution') -> 'Term':
        """Apply a substitution to this term."""
        return self


@dataclass
class Constant(Term):
    """A constant term (e.g., alice, bob)."""
    pass


@dataclass
class Variable(Term):
    """A variable term (e.g., X, Y)."""

    def apply_substitution(self, subst: 'Substitution') -> Term:
        """Apply substitution - replace variable if bound."""
        if self.name in subst.bindings:
            return subst.bindings[self.name]
        return self


class Substitution:
    """
    Represents variable bindings for unification.

    A substitution is a mapping from variables to terms.
    Example: {X: alice, Y: bob}

    Args:
        bindings: Dictionary mapping variable names to terms

    Example:
        >>> subst = Substitution({"X": Constant("alice")})
        >>> subst.get("X")
        Constant(name='alice')
        >>> subst.bind("Y", Constant("bob"))
        >>> print(subst)
        {X: alice, Y: bob}
    """

    def __init__(self, bindings: Optional[dict[str, Term]] = None):
        """Initialize a substitution.

        Args:
            bindings: Initial variable bindings (default: empty)
        """
        self.bindings: dict[str, Term] = bindings or {}

    def get(self, var_name: str) -> Optional[Term]:
        """Get the binding for a variable.

        Args:
            var_name: Variable name

        Returns:
            The term bound to this variable, or None if unbound
        """
        return self.bindings.get(var_name)

    def bind(self, var_name: str, term: Term) -> bool:
        """Add a new binding.

        Args:
            var_name: Variable name
            term: Term to bind

        Returns:
            True if binding succeeded, False if conflict with existing binding
        """
        if var_name in self.bindings:
            # Check if consistent with existing binding
            existing = self.bindings[var_name]
            if existing.name != term.name:
                return False  # Conflict!

        self.bindings[var_name] = term
        return True

    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose this substitution with another.

        Returns a new substitution that applies both.
        Formula: (σ ∘ θ)(X) = θ(σ(X))

        Args:
            other: Substitution to compose with

        Returns:
            Composed substitution
        """
        # Apply other to all our bindings
        new_bindings = {}
        for var, term in self.bindings.items():
            new_bindings[var] = term.apply_substitution(other)

        # Add bindings from other that aren't in self
        for var, term in other.bindings.items():
            if var not in new_bindings:
                new_bindings[var] = term

        return Substitution(new_bindings)

    def is_empty(self) -> bool:
        """Check if this is the empty substitution."""
        return len(self.bindings) == 0

    def __len__(self) -> int:
        """Return the number of bindings."""
        return len(self.bindings)

    def __contains__(self, var_name: str) -> bool:
        """Check if a variable is bound."""
        return var_name in self.bindings

    def __repr__(self) -> str:
        """String representation."""
        if not self.bindings:
            return "{}"
        items = [f"{var}: {term.name}" for var, term in self.bindings.items()]
        return "{" + ", ".join(items) + "}"

    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, Substitution):
            return False
        return self.bindings == other.bindings
