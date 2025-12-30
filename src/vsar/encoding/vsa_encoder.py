"""VSA encoder using shift-based positional encoding."""

import jax.numpy as jnp

from vsar.encoding.base import AtomEncoder
from vsar.kernel.base import KernelBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class VSAEncoder(AtomEncoder):
    """
    VSA encoder using shift-based positional encoding.

    Encodes atoms using circular shifts for position encoding:
        enc(p(t1,...,tk)) = shift(hv(t1), 1) ⊕ shift(hv(t2), 2) ⊕ ... ⊕ shift(hv(tk), k)

    Where:
    - hv(ti) is the entity hypervector from ENTITIES space
    - shift(vec, n) is circular permutation by n positions
    - ⊕ is the bundle operation (element-wise sum + normalization)

    Note: Predicate is NOT encoded in the vector - relies on predicate partitioning
    in the KB for separation.

    This approach avoids bind/unbind operations which are broken in vsax's FHRR
    implementation. Shift encoding is perfectly invertible: shift(shift(v,n),-n) = v.

    Args:
        backend: Kernel backend for hypervector operations
        registry: Symbol registry for looking up hypervectors
        seed: Random seed (unused, kept for API compatibility)

    Example:
        >>> from vsar.kernel.vsa_backend import FHRRBackend
        >>> from vsar.symbols.registry import SymbolRegistry
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> registry = SymbolRegistry(backend, seed=42)
        >>> encoder = VSAEncoder(backend, registry, seed=42)
        >>>
        >>> # Register symbols
        >>> registry.register(SymbolSpace.ENTITIES, "alice")
        >>> registry.register(SymbolSpace.ENTITIES, "bob")
        >>>
        >>> # Encode ground atom: parent(alice, bob)
        >>> atom_vec = encoder.encode_atom("parent", ["alice", "bob"])
        >>>
        >>> # Encode query: parent(alice, X)
        >>> query_vec = encoder.encode_query("parent", ["alice", None])
    """

    def __init__(self, backend: KernelBackend, registry: SymbolRegistry, seed: int = 42):
        super().__init__(backend, registry, seed)

    def encode_atom(self, predicate: str, args: list[str]) -> jnp.ndarray:
        """
        Encode a ground atom into a hypervector.

        Uses shift-based positional encoding where each argument is shifted
        by its position index.

        Args:
            predicate: Predicate name (e.g., "parent") - not encoded in vector
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
        if not args:
            raise ValueError("Arguments list cannot be empty")

        # Encode each argument with shift by position
        shifted_args = []
        for i, arg in enumerate(args):
            position = i + 1  # 1-indexed positions
            entity_vec = self.registry.register(SymbolSpace.ENTITIES, arg)

            # Shift entity by position: shift(entity, position)
            shifted = self.backend.permute(entity_vec, position)
            shifted_args.append(shifted)

        # Bundle all shifted arguments: shift(t1,1) ⊕ shift(t2,2) ⊕ ...
        atom_vec = self.backend.bundle(shifted_args)

        return self.backend.normalize(atom_vec)

    def encode_query(self, predicate: str, args: list[str | None]) -> jnp.ndarray:
        """
        Encode a query pattern into a hypervector.

        Query patterns have variables represented as None. Only bound
        arguments are included in the encoding using shift-based encoding.

        Args:
            predicate: Predicate name (e.g., "parent") - not encoded in vector
            args: List of entity names or None for variables
                  (e.g., ["alice", None] for parent(alice, X))

        Returns:
            Hypervector encoding of the query pattern (shifted bound args only)

        Raises:
            ValueError: If args list is empty
            ValueError: If all args are None (no bound variables)

        Example:
            >>> # Query: parent(alice, X)
            >>> query_vec = encoder.encode_query("parent", ["alice", None])
            >>> # Query: parent(X, bob)
            >>> query_vec = encoder.encode_query("parent", [None, "bob"])
        """
        if not args:
            raise ValueError("Arguments list cannot be empty")

        # Check that at least one argument is bound
        if all(arg is None for arg in args):
            raise ValueError("At least one argument must be bound (not None)")

        # Encode only bound arguments with shift by position
        shifted_args = []
        for i, arg in enumerate(args):
            if arg is not None:  # Skip variables
                position = i + 1  # 1-indexed positions
                entity_vec = self.registry.register(SymbolSpace.ENTITIES, arg)

                # Shift entity by position: shift(entity, position)
                shifted = self.backend.permute(entity_vec, position)
                shifted_args.append(shifted)

        # Bundle all shifted arguments
        query_vec = self.backend.bundle(shifted_args)

        return self.backend.normalize(query_vec)
