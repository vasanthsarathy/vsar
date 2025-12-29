"""Symbol registry for VSAR - manages symbol-to-hypervector mappings."""

from pathlib import Path
from typing import Optional

import jax.numpy as jnp

from vsar.kernel.base import KernelBackend

from .basis import generate_basis, load_basis, save_basis
from .spaces import SymbolSpace


class SymbolRegistry:
    """
    Central registry for all symbols in VSAR.

    The registry maintains mappings from (space, name) to hypervectors and
    provides cleanup functionality (reverse lookup via similarity search).

    Args:
        backend: Kernel backend for vector operations
        seed: Random seed for deterministic basis generation

    Example:
        >>> from vsar.kernel import FHRRBackend
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> registry = SymbolRegistry(backend, seed=42)
        >>> alice = registry.register(SymbolSpace.ENTITIES, "alice")
        >>> bob = registry.register(SymbolSpace.ENTITIES, "bob")
        >>> # Later, cleanup a noisy vector to find nearest symbol
        >>> nearest = registry.cleanup(SymbolSpace.ENTITIES, noisy_vec, k=1)
    """

    def __init__(self, backend: KernelBackend, seed: int = 42):
        self.backend = backend
        self.seed = seed
        self._basis: dict[tuple[SymbolSpace, str], jnp.ndarray] = {}

    def register(self, space: SymbolSpace, name: str) -> jnp.ndarray:
        """
        Register a symbol and get its hypervector.

        If the symbol is already registered, return its existing hypervector.
        Otherwise, generate a new deterministic basis vector.

        Args:
            space: Symbol space (E, R, A, etc.)
            name: Symbol name

        Returns:
            Hypervector for the symbol

        Example:
            >>> vec = registry.register(SymbolSpace.ENTITIES, "alice")
            >>> vec2 = registry.register(SymbolSpace.ENTITIES, "alice")
            >>> assert jnp.allclose(vec, vec2)  # Same symbol, same vector
        """
        key = (space, name)

        if key not in self._basis:
            # Generate new basis vector
            vec = generate_basis(space, name, self.backend, self.seed)
            self._basis[key] = vec

        return self._basis[key]

    def get(self, space: SymbolSpace, name: str) -> Optional[jnp.ndarray]:
        """
        Get hypervector for a symbol if it exists.

        Args:
            space: Symbol space
            name: Symbol name

        Returns:
            Hypervector if symbol is registered, None otherwise

        Example:
            >>> vec = registry.get(SymbolSpace.ENTITIES, "alice")
            >>> if vec is not None:
            ...     print("Symbol exists")
        """
        return self._basis.get((space, name))

    def cleanup(
        self, space: SymbolSpace, vector: jnp.ndarray, k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Find top-k nearest symbols in a space via similarity search.

        This is the "cleanup memory" operation - given a noisy or composite
        vector, find the most similar symbols in the specified space.

        Args:
            space: Symbol space to search within
            vector: Query hypervector
            k: Number of nearest symbols to return

        Returns:
            List of (symbol_name, similarity_score) tuples, sorted by similarity

        Example:
            >>> # After encoding and operations, cleanup to find nearest entity
            >>> results = registry.cleanup(SymbolSpace.ENTITIES, noisy_vec, k=5)
            >>> best_match, score = results[0]
            >>> print(f"Most likely symbol: {best_match} (score: {score:.3f})")
        """
        # Get all symbols in the specified space
        space_symbols = [
            (name, vec) for (s, name), vec in self._basis.items() if s == space
        ]

        if not space_symbols:
            return []

        # Compute similarities
        similarities: list[tuple[str, float]] = []
        for name, basis_vec in space_symbols:
            sim = self.backend.similarity(vector, basis_vec)
            similarities.append((name, sim))

        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def symbols(self, space: Optional[SymbolSpace] = None) -> list[str]:
        """
        List all registered symbols, optionally filtered by space.

        Args:
            space: If provided, only return symbols in this space

        Returns:
            List of symbol names

        Example:
            >>> all_entities = registry.symbols(SymbolSpace.ENTITIES)
            >>> all_symbols = registry.symbols()  # All spaces
        """
        if space is None:
            return [name for (_, name) in self._basis.keys()]
        else:
            return [name for (s, name) in self._basis.keys() if s == space]

    def count(self, space: Optional[SymbolSpace] = None) -> int:
        """
        Count registered symbols, optionally filtered by space.

        Args:
            space: If provided, only count symbols in this space

        Returns:
            Number of symbols

        Example:
            >>> num_entities = registry.count(SymbolSpace.ENTITIES)
            >>> total_symbols = registry.count()
        """
        if space is None:
            return len(self._basis)
        else:
            return sum(1 for (s, _) in self._basis.keys() if s == space)

    def save(self, path: Path) -> None:
        """
        Save registry to HDF5 file.

        Args:
            path: Path to save basis file

        Example:
            >>> registry.save(Path("vsar_basis.h5"))
        """
        save_basis(path, self._basis)

    def load(self, path: Path) -> None:
        """
        Load registry from HDF5 file.

        This replaces the current basis with the loaded one.

        Args:
            path: Path to basis file

        Example:
            >>> registry.load(Path("vsar_basis.h5"))
        """
        self._basis = load_basis(path)

    def clear(self) -> None:
        """
        Clear all registered symbols.

        Example:
            >>> registry.clear()
            >>> assert registry.count() == 0
        """
        self._basis.clear()
