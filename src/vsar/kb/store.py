"""Knowledge base storage with predicate partitioning."""

from typing import Any

import jax.numpy as jnp

from vsar.kernel.base import KernelBackend


class KnowledgeBase:
    """
    Predicate-partitioned knowledge base storage.

    Stores ground atoms as hypervectors bundled by predicate. Each predicate
    maintains a separate bundle to reduce noise during retrieval.

    Storage structure:
    - bundles: dict[predicate_name, bundled_hypervector]
    - facts: dict[predicate_name, list[fact_tuples]]

    Args:
        backend: Kernel backend for bundling operations

    Example:
        >>> from vsar.kernel.vsa_backend import FHRRBackend
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> kb = KnowledgeBase(backend)
        >>>
        >>> # Insert facts
        >>> kb.insert("parent", parent_alice_bob_vec, ("alice", "bob"))
        >>> kb.insert("parent", parent_bob_carol_vec, ("bob", "carol"))
        >>>
        >>> # Retrieve predicate bundle
        >>> bundle = kb.get_bundle("parent")
        >>> kb.count("parent")
        2
    """

    def __init__(self, backend: KernelBackend):
        self.backend = backend
        self._bundles: dict[str, jnp.ndarray] = {}
        self._facts: dict[str, list[tuple[Any, ...]]] = {}

    def insert(
        self, predicate: str, atom_vector: jnp.ndarray, fact: tuple[Any, ...]
    ) -> None:
        """
        Insert a ground atom into the knowledge base.

        The atom vector is bundled with existing atoms for this predicate.
        The fact tuple is stored for later reference.

        Args:
            predicate: Predicate name (e.g., "parent")
            atom_vector: Encoded hypervector for the atom
            fact: Tuple of arguments (e.g., ("alice", "bob"))

        Example:
            >>> kb.insert("parent", atom_vec, ("alice", "bob"))
            >>> kb.count("parent")
            1
        """
        # Initialize predicate storage if needed
        if predicate not in self._bundles:
            self._bundles[predicate] = atom_vector
            self._facts[predicate] = [fact]
        else:
            # Bundle new atom with existing bundle
            existing_bundle = self._bundles[predicate]
            new_bundle = self.backend.bundle([existing_bundle, atom_vector])
            self._bundles[predicate] = new_bundle
            self._facts[predicate].append(fact)

    def get_bundle(self, predicate: str) -> jnp.ndarray | None:
        """
        Get the bundled hypervector for a predicate.

        Args:
            predicate: Predicate name

        Returns:
            Bundled hypervector for all atoms of this predicate,
            or None if predicate not found

        Example:
            >>> bundle = kb.get_bundle("parent")
            >>> bundle.shape
            (512,)
        """
        return self._bundles.get(predicate)

    def get_facts(self, predicate: str) -> list[tuple[Any, ...]]:
        """
        Get all facts for a predicate.

        Args:
            predicate: Predicate name

        Returns:
            List of fact tuples, or empty list if predicate not found

        Example:
            >>> facts = kb.get_facts("parent")
            >>> facts
            [("alice", "bob"), ("bob", "carol")]
        """
        return self._facts.get(predicate, [])

    def predicates(self) -> list[str]:
        """
        Get list of all predicates in the KB.

        Returns:
            List of predicate names

        Example:
            >>> kb.predicates()
            ["parent", "sibling"]
        """
        return list(self._bundles.keys())

    def count(self, predicate: str | None = None) -> int:
        """
        Count facts in the KB.

        Args:
            predicate: Optional predicate name to count facts for.
                      If None, returns total fact count across all predicates.

        Returns:
            Number of facts

        Example:
            >>> kb.count("parent")
            2
            >>> kb.count()
            5
        """
        if predicate is None:
            return sum(len(facts) for facts in self._facts.values())
        return len(self._facts.get(predicate, []))

    def has_predicate(self, predicate: str) -> bool:
        """
        Check if a predicate exists in the KB.

        Args:
            predicate: Predicate name

        Returns:
            True if predicate has facts, False otherwise

        Example:
            >>> kb.has_predicate("parent")
            True
            >>> kb.has_predicate("nonexistent")
            False
        """
        return predicate in self._bundles

    def clear(self) -> None:
        """
        Clear all facts from the KB.

        Example:
            >>> kb.clear()
            >>> kb.count()
            0
        """
        self._bundles.clear()
        self._facts.clear()

    def clear_predicate(self, predicate: str) -> None:
        """
        Clear all facts for a specific predicate.

        Args:
            predicate: Predicate name

        Example:
            >>> kb.clear_predicate("parent")
            >>> kb.has_predicate("parent")
            False
        """
        if predicate in self._bundles:
            del self._bundles[predicate]
            del self._facts[predicate]
