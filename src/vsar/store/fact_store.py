"""Fact store with predicate indexing and paraconsistent tracking."""

from typing import Optional
import jax.numpy as jnp

from ..kernel.base import KernelBackend
from .item import Item, ItemKind
from .belief import BeliefState, Literal


class FactStore:
    """
    Storage for facts with predicate indexing and belief tracking.

    The FactStore maintains:
    - Predicate-indexed facts for efficient retrieval
    - Paraconsistent belief states for literals
    - Similarity-based retrieval for approximate matching

    Args:
        backend: VSA backend for similarity computations

    Example:
        >>> from vsar.kernel.vsa_backend import FHRRBackend
        >>> backend = FHRRBackend(dim=2048, seed=42)
        >>> store = FactStore(backend)
        >>>
        >>> # Insert a fact
        >>> literal = Literal("parent", ("alice", "bob"))
        >>> item = Item(vec=encoded_vec, kind=ItemKind.FACT, weight=1.0)
        >>> store.insert(item, literal)
        >>>
        >>> # Retrieve facts
        >>> facts = store.retrieve_by_predicate("parent")
        >>> belief = store.get_belief(literal)
    """

    def __init__(self, backend: KernelBackend):
        """Initialize the fact store.

        Args:
            backend: VSA backend for similarity operations
        """
        self.backend = backend
        self._by_predicate: dict[str, list[Item]] = {}
        self._belief_state: dict[str, BeliefState] = {}
        self._all_items: list[Item] = []

    def insert(self, item: Item, literal: Literal):
        """
        Insert a fact and update belief state.

        Args:
            item: The item to insert
            literal: The literal this item represents

        Example:
            >>> literal = Literal("parent", ("alice", "bob"))
            >>> item = Item(vec=vec, kind=ItemKind.FACT, weight=0.9)
            >>> store.insert(item, literal)
        """
        # Validate that this is a fact
        if not item.is_fact():
            raise ValueError(f"Can only insert facts, got {item.kind}")

        # Add to predicate index
        pred = literal.predicate
        if pred not in self._by_predicate:
            self._by_predicate[pred] = []
        self._by_predicate[pred].append(item)

        # Add to all items list
        self._all_items.append(item)

        # Update belief state
        # Use non-negated literal as key so L and ¬L share same BeliefState
        base_literal = literal if not literal.negated else literal.negate()
        key = base_literal.to_key()
        if key not in self._belief_state:
            self._belief_state[key] = BeliefState()

        if literal.negated:
            # Support for ¬L
            self._belief_state[key] = self._belief_state[key].update_negative(item.weight)
        else:
            # Support for L
            self._belief_state[key] = self._belief_state[key].update_positive(item.weight)

    def retrieve_by_predicate(self, predicate: str) -> list[Item]:
        """
        Retrieve all facts with a given predicate.

        Args:
            predicate: Predicate name to search for

        Returns:
            List of items with that predicate

        Example:
            >>> facts = store.retrieve_by_predicate("parent")
            >>> len(facts)
            2
        """
        return self._by_predicate.get(predicate, [])

    def retrieve_similar(
        self,
        query_vec: jnp.ndarray,
        k: int = 10,
        threshold: float = 0.0,
        predicate: Optional[str] = None
    ) -> list[tuple[Item, float]]:
        """
        Retrieve top-k most similar facts via exploratory search.

        Args:
            query_vec: Query hypervector
            k: Number of results to return
            threshold: Minimum similarity threshold
            predicate: Optional predicate to filter by

        Returns:
            List of (item, similarity) tuples, sorted by similarity (descending)

        Example:
            >>> results = store.retrieve_similar(query_vec, k=5, threshold=0.5)
            >>> for item, sim in results:
            ...     print(f"Similarity: {sim:.2f}")
        """
        # Get items to search over
        if predicate is not None:
            candidates = self._by_predicate.get(predicate, [])
        else:
            candidates = self._all_items

        # Compute similarities
        results = []
        for item in candidates:
            sim = float(self.backend.similarity(query_vec, item.vec))
            if sim >= threshold:
                results.append((item, sim))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return results[:k]

    def check_novelty(
        self,
        query_vec: jnp.ndarray,
        threshold: float = 0.9,
        predicate: Optional[str] = None
    ) -> bool:
        """
        Check if a fact is novel (not already present).

        A fact is considered novel if no existing fact has similarity
        above the threshold.

        Args:
            query_vec: Query hypervector
            threshold: Similarity threshold for considering facts as duplicates
            predicate: Optional predicate to filter by

        Returns:
            True if novel, False if similar fact exists

        Example:
            >>> is_novel = store.check_novelty(new_fact_vec, threshold=0.95)
            >>> if is_novel:
            ...     store.insert(item, literal)
        """
        similar = self.retrieve_similar(
            query_vec,
            k=1,
            threshold=threshold,
            predicate=predicate
        )
        return len(similar) == 0

    def get_belief(self, literal: Literal) -> BeliefState:
        """
        Get the paraconsistent belief state for a literal.

        Args:
            literal: The literal to query

        Returns:
            BeliefState with support for L and ¬L

        Example:
            >>> literal = Literal("parent", ("alice", "bob"))
            >>> belief = store.get_belief(literal)
            >>> if belief.is_consistent():
            ...     print("Belief is consistent")
        """
        # Use non-negated literal as key
        base_literal = literal if not literal.negated else literal.negate()
        key = base_literal.to_key()
        return self._belief_state.get(key, BeliefState())

    def predicates(self) -> list[str]:
        """Get list of all predicates in the store."""
        return list(self._by_predicate.keys())

    def __len__(self) -> int:
        """Return the total number of facts."""
        return len(self._all_items)

    def __repr__(self) -> str:
        """String representation."""
        num_preds = len(self._by_predicate)
        num_beliefs = len(self._belief_state)
        return f"FactStore({len(self)} facts, {num_preds} predicates, {num_beliefs} beliefs)"
