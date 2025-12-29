"""Abstract encoder interface for atom encoding."""

from abc import ABC, abstractmethod

import jax.numpy as jnp

from vsar.kernel.base import KernelBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class AtomEncoder(ABC):
    """
    Abstract interface for encoding ground atoms and query patterns.

    An atom encoder transforms logical atoms (predicates with arguments)
    into hypervector representations. It supports both:
    - Ground atoms: parent(alice, bob) - all arguments bound
    - Query patterns: parent(alice, X) - some arguments are variables (None)

    Args:
        backend: Kernel backend for hypervector operations
        registry: Symbol registry for looking up hypervectors
        seed: Random seed for deterministic role vector generation

    Example:
        >>> encoder = VSAEncoder(backend, registry, seed=42)
        >>> # Encode ground atom
        >>> atom_vec = encoder.encode_atom("parent", ["alice", "bob"])
        >>> # Encode query pattern
        >>> query_vec = encoder.encode_query("parent", ["alice", None])
    """

    def __init__(
        self, backend: KernelBackend, registry: SymbolRegistry, seed: int = 42
    ):
        self.backend = backend
        self.registry = registry
        self.seed = seed

    @abstractmethod
    def encode_atom(self, predicate: str, args: list[str]) -> jnp.ndarray:
        """
        Encode a ground atom into a hypervector.

        Args:
            predicate: Predicate name (e.g., "parent")
            args: List of entity names (e.g., ["alice", "bob"])

        Returns:
            Hypervector encoding of the atom

        Raises:
            ValueError: If args list is empty

        Example:
            >>> vec = encoder.encode_atom("parent", ["alice", "bob"])
            >>> vec.shape
            (512,)
        """
        pass

    @abstractmethod
    def encode_query(
        self, predicate: str, args: list[str | None]
    ) -> jnp.ndarray:
        """
        Encode a query pattern into a hypervector.

        Query patterns have variables represented as None. For example,
        `parent(alice, X)` is encoded as `["alice", None]`.

        Args:
            predicate: Predicate name (e.g., "parent")
            args: List of entity names or None for variables
                  (e.g., ["alice", None] for parent(alice, X))

        Returns:
            Hypervector encoding of the query pattern

        Raises:
            ValueError: If args list is empty
            ValueError: If all args are None (no bound variables)

        Example:
            >>> # Query: parent(alice, X)
            >>> query_vec = encoder.encode_query("parent", ["alice", None])
            >>> # Query: parent(X, bob)
            >>> query_vec = encoder.encode_query("parent", [None, "bob"])
        """
        pass

    def get_variable_positions(self, args: list[str | None]) -> list[int]:
        """
        Get positions (1-indexed) of variables in the argument list.

        Args:
            args: List of entity names or None for variables

        Returns:
            List of 1-indexed positions where variables occur

        Example:
            >>> encoder.get_variable_positions(["alice", None, "bob"])
            [2]
            >>> encoder.get_variable_positions([None, None, "bob"])
            [1, 2]
        """
        return [i + 1 for i, arg in enumerate(args) if arg is None]

    def get_bound_positions(self, args: list[str | None]) -> list[int]:
        """
        Get positions (1-indexed) of bound arguments in the argument list.

        Args:
            args: List of entity names or None for variables

        Returns:
            List of 1-indexed positions where bound arguments occur

        Example:
            >>> encoder.get_bound_positions(["alice", None, "bob"])
            [1, 3]
            >>> encoder.get_bound_positions([None, None, "bob"])
            [3]
        """
        return [i + 1 for i, arg in enumerate(args) if arg is not None]
