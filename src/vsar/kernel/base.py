"""Abstract base classes for VSAR kernel backends."""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class KernelBackend(ABC):
    """
    Abstract base for VSA and Clifford algebra backends.

    This interface defines the core operations that both VSA and Clifford
    backends must implement, enabling polymorphic use throughout VSAR.
    """

    @abstractmethod
    def bind(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Composition operation (⊗ for VSA, ⋆ for Clifford).

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Bound/composed hypervector

        Example:
            >>> backend = FHRRBackend(dim=512, seed=42)
            >>> a = backend.generate_random(jax.random.PRNGKey(0), (512,))
            >>> b = backend.generate_random(jax.random.PRNGKey(1), (512,))
            >>> c = backend.bind(a, b)
        """
        pass

    @abstractmethod
    def unbind(self, bound: jnp.ndarray, factor: jnp.ndarray) -> jnp.ndarray:
        """
        Factor isolation (approx inverse for VSA, geometric division for Clifford).

        Args:
            bound: Bound hypervector
            factor: Known factor to remove

        Returns:
            Isolated/unbound hypervector (approximate)

        Example:
            >>> c = backend.bind(a, b)
            >>> b_recovered = backend.unbind(c, a)
            >>> similarity = backend.similarity(b, b_recovered)  # Should be high
        """
        pass

    @abstractmethod
    def bundle(
        self, vectors: list[jnp.ndarray] | jnp.ndarray, axis: int = 0
    ) -> jnp.ndarray:
        """
        Superposition/bundling operation.

        Args:
            vectors: List of hypervectors or array of hypervectors
            axis: Axis along which to bundle (default: 0)

        Returns:
            Bundled hypervector

        Example:
            >>> vectors = [a, b, c]
            >>> bundled = backend.bundle(vectors)
        """
        pass

    @abstractmethod
    def similarity(self, a: jnp.ndarray, b: jnp.ndarray) -> float:
        """
        Similarity metric (cosine for VSA, depends on grade for Clifford).

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Similarity score in [0, 1] (or [-1, 1] for some metrics)

        Example:
            >>> sim = backend.similarity(a, b)
            >>> assert 0.0 <= sim <= 1.0
        """
        pass

    @abstractmethod
    def generate_random(
        self, key: jax.random.PRNGKey, shape: tuple[int, ...]
    ) -> jnp.ndarray:
        """
        Generate random basis vector.

        Args:
            key: JAX random key for reproducibility
            shape: Shape of the vector to generate

        Returns:
            Random hypervector

        Example:
            >>> key = jax.random.PRNGKey(42)
            >>> vec = backend.generate_random(key, (512,))
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Vector dimension.

        Returns:
            Dimension of hypervectors in this backend
        """
        pass

    @abstractmethod
    def normalize(self, vec: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize a hypervector.

        Args:
            vec: Hypervector to normalize

        Returns:
            Normalized hypervector

        Example:
            >>> vec = backend.generate_random(jax.random.PRNGKey(0), (512,))
            >>> normalized = backend.normalize(vec)
        """
        pass
