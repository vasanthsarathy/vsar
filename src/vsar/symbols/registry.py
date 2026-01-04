"""Symbol registry managing all typed codebooks for VSAR v2.0.

The SymbolRegistry provides a central interface for registering symbols
across all typed symbol spaces and performing typed cleanup operations.
"""

import jax.numpy as jnp
from typing import Optional

from .spaces import SymbolSpace
from .codebook import TypedCodebook


class SymbolRegistry:
    """
    Central registry managing typed codebooks for all symbol spaces.

    The SymbolRegistry creates and manages separate TypedCodebook instances
    for each symbol space, providing a unified interface for symbol registration
    and cleanup operations.

    Args:
        dim: Hypervector dimension for all codebooks
        seed: Base random seed (each codebook gets seed + space_index)

    Example:
        >>> registry = SymbolRegistry(dim=512, seed=42)
        >>>
        >>> # Register symbols in different spaces
        >>> alice = registry.register(SymbolSpace.ENTITIES, "alice")
        >>> parent = registry.register(SymbolSpace.PREDICATES, "parent")
        >>> arg1 = registry.register(SymbolSpace.ARG_ROLES, "ARG1")
        >>>
        >>> # Typed cleanup
        >>> results = registry.cleanup(SymbolSpace.ENTITIES, noisy_vec, k=3)
    """

    def __init__(self, dim: int = 512, seed: int = 42):
        """Initialize the symbol registry.

        Args:
            dim: Hypervector dimension for all codebooks
            seed: Base random seed for reproducibility
        """
        self.dim = dim
        self.seed = seed

        # Create a codebook for each symbol space
        self._codebooks: dict[SymbolSpace, TypedCodebook] = {}

        for i, space in enumerate(SymbolSpace):
            # Use different seed for each space to ensure orthogonality
            space_seed = seed + (i * 10000)
            self._codebooks[space] = TypedCodebook(space, dim=dim, seed=space_seed)

    def register(self, space: SymbolSpace, name: str) -> jnp.ndarray:
        """
        Register a symbol in a specific symbol space.

        Args:
            space: The symbol space for this symbol
            name: Symbol name to register

        Returns:
            The hypervector for this symbol

        Example:
            >>> registry = SymbolRegistry(dim=512)
            >>> alice = registry.register(SymbolSpace.ENTITIES, "alice")
            >>> parent = registry.register(SymbolSpace.PREDICATES, "parent")
        """
        return self._codebooks[space].register(name)

    def get(self, space: SymbolSpace, name: str) -> Optional[jnp.ndarray]:
        """
        Get the vector for a registered symbol.

        Args:
            space: The symbol space to search
            name: Symbol name

        Returns:
            The hypervector if registered, None otherwise

        Example:
            >>> registry = SymbolRegistry(dim=512)
            >>> registry.register(SymbolSpace.ENTITIES, "alice")
            >>> vec = registry.get(SymbolSpace.ENTITIES, "alice")
            >>> vec is not None
            True
        """
        return self._codebooks[space].get(name)

    def cleanup(
        self,
        space: SymbolSpace,
        vec: jnp.ndarray,
        k: int = 1,
        threshold: float = 0.0
    ) -> list[tuple[str, float]]:
        """
        Perform typed cleanup: find nearest symbols in a specific space.

        This is the key operation for structure-informed decoding. After unbinding
        to isolate a component, we perform typed cleanup to commit to a discrete symbol.

        Args:
            space: The symbol space to search
            vec: Query hypervector
            k: Number of nearest neighbors to return
            threshold: Minimum similarity threshold (0-1 range)

        Returns:
            List of (symbol_name, similarity) tuples, sorted by similarity

        Example:
            >>> registry = SymbolRegistry(dim=512)
            >>> registry.register(SymbolSpace.ENTITIES, "alice")
            >>> registry.register(SymbolSpace.ENTITIES, "bob")
            >>>
            >>> # Query with a vector similar to alice
            >>> results = registry.cleanup(SymbolSpace.ENTITIES, alice_vec, k=2)
            >>> results[0][0]  # Should be 'alice'
            'alice'
        """
        return self._codebooks[space].cleanup(vec, k=k, threshold=threshold)

    def contains(self, space: SymbolSpace, name: str) -> bool:
        """
        Check if a symbol is registered in a space.

        Args:
            space: The symbol space to check
            name: Symbol name

        Returns:
            True if the symbol is registered in this space

        Example:
            >>> registry = SymbolRegistry(dim=512)
            >>> registry.register(SymbolSpace.ENTITIES, "alice")
            >>> registry.contains(SymbolSpace.ENTITIES, "alice")
            True
            >>> registry.contains(SymbolSpace.PREDICATES, "alice")
            False
        """
        return name in self._codebooks[space]

    def get_codebook(self, space: SymbolSpace) -> TypedCodebook:
        """
        Get the codebook for a specific symbol space.

        Args:
            space: The symbol space

        Returns:
            The TypedCodebook for this space

        Example:
            >>> registry = SymbolRegistry(dim=512)
            >>> entities = registry.get_codebook(SymbolSpace.ENTITIES)
            >>> entities.register("alice")
        """
        return self._codebooks[space]

    def symbol_count(self, space: Optional[SymbolSpace] = None) -> int:
        """
        Get the number of registered symbols.

        Args:
            space: If provided, count symbols in this space only.
                   If None, count symbols across all spaces.

        Returns:
            Number of registered symbols

        Example:
            >>> registry = SymbolRegistry(dim=512)
            >>> registry.register(SymbolSpace.ENTITIES, "alice")
            >>> registry.register(SymbolSpace.PREDICATES, "parent")
            >>> registry.symbol_count()
            2
            >>> registry.symbol_count(SymbolSpace.ENTITIES)
            1
        """
        if space is not None:
            return len(self._codebooks[space])
        else:
            return sum(len(codebook) for codebook in self._codebooks.values())

    def __repr__(self) -> str:
        """Return a string representation."""
        total = self.symbol_count()
        return f"SymbolRegistry({total} symbols across {len(SymbolSpace)} spaces, dim={self.dim})"
