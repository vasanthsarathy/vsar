"""Role vector management for VSAR atom encoding."""

import jax
import jax.numpy as jnp

from vsar.kernel.base import KernelBackend


class RoleVectorManager:
    """
    Manages role vectors (ρ1, ρ2, ...) for argument positions.

    Role vectors are used in VSA encoding to distinguish between different
    argument positions in predicates. For example, in `parent(alice, bob)`,
    ρ1 is bound to "alice" and ρ2 is bound to "bob".

    The role vectors are:
    - Fixed per arity (same ρ1, ρ2 across all predicates)
    - Maximally dissimilar (orthogonal) to each other
    - Deterministically generated from a seed

    Args:
        backend: Kernel backend for vector generation
        seed: Random seed for deterministic generation

    Example:
        >>> from vsar.kernel import FHRRBackend
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> manager = RoleVectorManager(backend, seed=42)
        >>> role1 = manager.get_role(1)  # ρ1
        >>> role2 = manager.get_role(2)  # ρ2
    """

    def __init__(self, backend: KernelBackend, seed: int = 42):
        self.backend = backend
        self.seed = seed
        self._roles: dict[int, jnp.ndarray] = {}

    def get_role(self, position: int) -> jnp.ndarray:
        """
        Get role vector for argument position.

        Args:
            position: Argument position (1-indexed: 1, 2, 3, ...)

        Returns:
            Role hypervector for this position

        Raises:
            ValueError: If position is less than 1

        Example:
            >>> role1 = manager.get_role(1)  # First argument
            >>> role2 = manager.get_role(2)  # Second argument
        """
        if position < 1:
            raise ValueError(f"Position must be >= 1, got {position}")

        if position not in self._roles:
            # Generate deterministic role vector
            role_seed = self.seed + 10000 + position  # Offset to avoid collision with symbols
            key = jax.random.PRNGKey(role_seed)
            role_vec = self.backend.generate_random(key, (self.backend.dimension,))
            self._roles[position] = self.backend.normalize(role_vec)

        return self._roles[position]

    def get_roles(self, arity: int) -> list[jnp.ndarray]:
        """
        Get role vectors for all positions up to arity.

        Args:
            arity: Number of arguments

        Returns:
            List of role vectors [ρ1, ρ2, ..., ρarity]

        Example:
            >>> roles = manager.get_roles(3)  # For ternary predicate
            >>> assert len(roles) == 3
        """
        return [self.get_role(i) for i in range(1, arity + 1)]

    def clear(self) -> None:
        """
        Clear all cached role vectors.

        Example:
            >>> manager.clear()
            >>> assert len(manager._roles) == 0
        """
        self._roles.clear()

    def similarity_matrix(self, arity: int) -> jnp.ndarray:
        """
        Compute similarity matrix between role vectors.

        This is useful for verifying that role vectors are sufficiently
        dissimilar (approximately orthogonal).

        Args:
            arity: Number of roles to compare

        Returns:
            Similarity matrix of shape (arity, arity)

        Example:
            >>> sim_matrix = manager.similarity_matrix(3)
            >>> # Diagonal should be ~1.0, off-diagonal should be low
            >>> assert sim_matrix[0, 0] > 0.99
            >>> assert abs(sim_matrix[0, 1]) < 0.5  # Dissimilar
        """
        roles = self.get_roles(arity)
        n = len(roles)
        sim_matrix = jnp.zeros((n, n))

        for i in range(n):
            for j in range(n):
                sim = self.backend.similarity(roles[i], roles[j])
                sim_matrix = sim_matrix.at[i, j].set(sim)

        return sim_matrix
