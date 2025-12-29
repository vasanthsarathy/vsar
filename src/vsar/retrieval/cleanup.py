"""Cleanup memory for symbol decoding."""

import jax.numpy as jnp

from vsar.kernel.base import KernelBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


def cleanup(
    space: SymbolSpace,
    vector: jnp.ndarray,
    registry: SymbolRegistry,
    backend: KernelBackend,
    k: int = 10,
) -> list[tuple[str, float]]:
    """
    Find top-k nearest symbols to a noisy hypervector.

    Cleanup memory performs nearest neighbor search in the symbol space
    to decode a noisy hypervector back to symbolic names.

    Args:
        space: Symbol space to search in (e.g., ENTITIES, RELATIONS)
        vector: Noisy hypervector to decode
        registry: Symbol registry containing basis vectors
        backend: Kernel backend for similarity computation
        k: Number of top results to return

    Returns:
        List of (symbol_name, similarity_score) tuples, sorted by score descending

    Example:
        >>> # Noisy vector approximating "bob"
        >>> results = cleanup(
        ...     SymbolSpace.ENTITIES, noisy_vec, registry, backend, k=5
        ... )
        >>> results[0]  # Top match
        ('bob', 0.82)
    """
    return registry.cleanup(space, vector, k)


def batch_cleanup(
    space: SymbolSpace,
    vectors: list[jnp.ndarray],
    registry: SymbolRegistry,
    backend: KernelBackend,
    k: int = 10,
) -> list[list[tuple[str, float]]]:
    """
    Perform cleanup on multiple vectors.

    Args:
        space: Symbol space to search in
        vectors: List of noisy hypervectors to decode
        registry: Symbol registry
        backend: Kernel backend
        k: Number of top results per vector

    Returns:
        List of cleanup results, one per input vector

    Example:
        >>> # Cleanup multiple noisy vectors
        >>> results = batch_cleanup(
        ...     SymbolSpace.ENTITIES, [vec1, vec2, vec3], registry, backend, k=5
        ... )
        >>> len(results)
        3
    """
    return [cleanup(space, vec, registry, backend, k) for vec in vectors]


def get_top_symbol(
    space: SymbolSpace,
    vector: jnp.ndarray,
    registry: SymbolRegistry,
    backend: KernelBackend,
) -> tuple[str, float] | None:
    """
    Get the single best matching symbol.

    Convenience function for retrieving just the top-1 result.

    Args:
        space: Symbol space to search in
        vector: Noisy hypervector to decode
        registry: Symbol registry
        backend: Kernel backend

    Returns:
        (symbol_name, similarity_score) tuple, or None if no symbols in space

    Example:
        >>> result = get_top_symbol(
        ...     SymbolSpace.ENTITIES, noisy_vec, registry, backend
        ... )
        >>> result
        ('bob', 0.82)
    """
    results = cleanup(space, vector, registry, backend, k=1)
    return results[0] if results else None
