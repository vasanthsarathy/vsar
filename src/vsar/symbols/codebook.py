"""Typed codebook implementation for VSAR v2.0.

This module provides TypedCodebook for managing symbols within a single
typed symbol space with cleanup operations.
"""

import jax
import jax.numpy as jnp
import vsax
from typing import Optional

from .spaces import SymbolSpace


class TypedCodebook:
    """
    A codebook for a single typed symbol space.

    TypedCodebook maintains a mapping from symbol names to hypervectors
    within a specific symbol space. It uses VSAX's memory for generating
    random basis vectors and performing cleanup (nearest neighbor search).

    Args:
        space: The symbol space this codebook represents
        dim: Hypervector dimension
        seed: Random seed for reproducible basis vectors

    Example:
        >>> codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512, seed=42)
        >>> alice_vec = codebook.register("alice")
        >>> bob_vec = codebook.register("bob")
        >>>
        >>> # Later, cleanup a noisy vector
        >>> candidates = codebook.cleanup(noisy_vec, k=3)
        >>> # Returns: [("alice", 0.95), ("bob", 0.12), ...]
    """

    def __init__(self, space: SymbolSpace, dim: int = 512, seed: int = 42):
        """Initialize a typed codebook.

        Args:
            space: Symbol space this codebook represents
            dim: Hypervector dimension
            seed: Random seed for basis vector generation
        """
        self.space = space
        self.dim = dim
        self.seed = seed

        # Track registered symbols
        self._symbols: dict[str, jnp.ndarray] = {}

        # Counter for generating unique seeds per symbol
        self._symbol_count = 0

    def register(self, name: str) -> jnp.ndarray:
        """
        Register a symbol and return its basis vector.

        If the symbol is already registered, returns the existing vector.
        Otherwise, generates a new random basis vector for this symbol.

        Args:
            name: Symbol name to register

        Returns:
            The hypervector for this symbol

        Example:
            >>> codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512)
            >>> alice = codebook.register("alice")
            >>> alice.shape
            (512,)
        """
        if name in self._symbols:
            return self._symbols[name]

        # Generate unique seed for this symbol
        symbol_seed = self.seed + self._symbol_count
        self._symbol_count += 1

        # Generate random basis vector
        key = jax.random.PRNGKey(symbol_seed)
        vec = vsax.sample_complex_random(dim=self.dim, n=1, key=key).squeeze()

        # Normalize
        vec = vec / jnp.linalg.norm(vec)

        # Store in symbol dict
        self._symbols[name] = vec

        return vec

    def get(self, name: str) -> Optional[jnp.ndarray]:
        """
        Get the vector for a registered symbol.

        Args:
            name: Symbol name

        Returns:
            The hypervector if registered, None otherwise

        Example:
            >>> codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512)
            >>> codebook.register("alice")
            >>> vec = codebook.get("alice")
            >>> vec is not None
            True
        """
        return self._symbols.get(name)

    def cleanup(self, vec: jnp.ndarray, k: int = 1, threshold: float = 0.0) -> list[tuple[str, float]]:
        """
        Find the k nearest symbols to a query vector (typed cleanup).

        This performs nearest neighbor search within this symbol space only,
        returning the k most similar registered symbols with their similarity scores.

        Args:
            vec: Query hypervector
            k: Number of nearest neighbors to return
            threshold: Minimum similarity threshold (0-1 range, after normalization)

        Returns:
            List of (symbol_name, similarity) tuples, sorted by similarity (highest first)

        Example:
            >>> codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512)
            >>> alice = codebook.register("alice")
            >>>
            >>> # Cleanup exact match
            >>> results = codebook.cleanup(alice, k=1)
            >>> results[0][0]  # name
            'alice'
            >>> results[0][1]  # similarity
            1.0
        """
        if len(self._symbols) == 0:
            return []

        # Normalize query vector
        vec_norm = vec / jnp.linalg.norm(vec)

        # Compute similarities to all registered symbols
        candidates = []
        for name, symbol_vec in self._symbols.items():
            # Compute cosine similarity (complex dot product)
            similarity = float(jnp.abs(jnp.sum(vec_norm * jnp.conj(symbol_vec)) / jnp.linalg.norm(symbol_vec)))
            candidates.append((name, similarity))

        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top-k and filter by threshold
        results = []
        for name, sim in candidates[:k]:
            if sim >= threshold:
                results.append((name, sim))

        return results

    def __len__(self) -> int:
        """Return the number of registered symbols."""
        return len(self._symbols)

    def __contains__(self, name: str) -> bool:
        """Check if a symbol is registered."""
        return name in self._symbols

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"TypedCodebook({self.space}, {len(self)} symbols, dim={self.dim})"
