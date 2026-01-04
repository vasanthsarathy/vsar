"""Unification for pattern matching."""

from typing import Optional
import jax.numpy as jnp

from ..kernel.base import KernelBackend
from ..symbols.registry import SymbolRegistry
from .decoder import Atom
from .substitution import Substitution, Term, Constant, Variable


class Unifier:
    """
    Unifier for pattern matching via VSA decoding.

    This class implements unification of atoms by:
    1. Checking structural compatibility
    2. Building substitutions for variables
    3. Verifying decoded values match

    Args:
        backend: VSA backend for operations
        registry: Symbol registry

    Example:
        >>> unifier = Unifier(backend, registry)
        >>> atom1 = Atom("parent", [Constant("alice"), Variable("X")])
        >>> atom2 = Atom("parent", [Constant("alice"), Constant("bob")])
        >>> subst = unifier.unify(atom1, atom2)
        >>> print(subst)  # {X: bob}
    """

    def __init__(self, backend: KernelBackend, registry: SymbolRegistry):
        """Initialize the unifier.

        Args:
            backend: VSA backend
            registry: Symbol registry
        """
        self.backend = backend
        self.registry = registry

    def unify(self, atom1: Atom, atom2: Atom) -> Optional[Substitution]:
        """
        Unify two atoms, returning a substitution if successful.

        Args:
            atom1: First atom
            atom2: Second atom

        Returns:
            Substitution mapping variables to terms, or None if unification fails
        """
        # Check predicates match
        if atom1.predicate != atom2.predicate:
            return None

        # Check arity matches
        if len(atom1.args) != len(atom2.args):
            return None

        # Build substitution
        subst = Substitution()
        for arg1, arg2 in zip(atom1.args, atom2.args):
            if not self._unify_terms(arg1, arg2, subst):
                return None

        return subst

    def _unify_terms(self, term1: Term, term2: Term, subst: Substitution) -> bool:
        """
        Unify two terms, updating the substitution.

        Args:
            term1: First term
            term2: Second term
            subst: Substitution to update

        Returns:
            True if unification succeeds, False otherwise
        """
        # Apply existing substitution
        term1 = term1.apply_substitution(subst)
        term2 = term2.apply_substitution(subst)

        # Variable-variable
        if isinstance(term1, Variable) and isinstance(term2, Variable):
            if term1.name == term2.name:
                return True  # Same variable
            # Bind one to the other
            return subst.bind(term1.name, term2)

        # Variable-constant
        if isinstance(term1, Variable):
            return subst.bind(term1.name, term2)
        if isinstance(term2, Variable):
            return subst.bind(term2.name, term1)

        # Constant-constant
        if isinstance(term1, Constant) and isinstance(term2, Constant):
            return term1.name == term2.name

        # Unknown term types
        return False

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"Unifier(backend={self.backend.__class__.__name__})"
