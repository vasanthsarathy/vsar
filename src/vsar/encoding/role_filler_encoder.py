"""Hybrid encoder combining predicate binding with shift-based positional encoding.

This encoder implements the expert-recommended hybrid approach:
    enc(p(t1,...,tk)) = P_p ⊗ (shift(enc(t1), 1) ⊕ shift(enc(t2), 2) ⊕ ... ⊕ shift(enc(tk), k))

Where:
- P_p is the predicate vector from PREDICATES symbol space
- shift(v, n) is circular permutation by n positions
- enc(ti) are entity vectors from the ENTITIES space
- ⊗ is the bind operation (element-wise multiplication for FHRR)
- ⊕ is the bundle operation (element-wise sum + normalization)

This approach combines:
- Predicate distinguishability (from binding P_p)
- Clean invertible positional encoding (from shift operations)
- Low cross-talk between argument positions

To decode position i:
1. unbind(fact_vec, P_p) → args_bundle
2. shift(args_bundle, -i) → entity_vec
3. cleanup in ENTITIES space
"""

import jax.numpy as jnp

from vsar.encoding.base import AtomEncoder
from vsar.kernel.base import KernelBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class RoleFillerEncoder(AtomEncoder):
    """
    Role-filler encoder using bind operations for positional encoding.

    Encodes atoms by binding each argument with its role vector:
        enc(p(t1,...,tk)) = ARG1 ⊗ hv(t1) ⊕ ARG2 ⊗ hv(t2) ⊕ ... ⊕ ARGk ⊗ hv(tk)

    Where:
    - hv(ti) is the entity hypervector from ENTITIES space
    - ARGi is the role vector for position i from ARG_ROLES space
    - ⊗ is the bind operation
    - ⊕ is the bundle operation (element-wise sum + normalization)

    Note: Predicate is NOT encoded in the vector - relies on predicate partitioning
    in the KB for separation.

    This encoding is invertible: unbind(enc, ARGi) ≈ hv(ti)

    Args:
        backend: Kernel backend for hypervector operations
        registry: Symbol registry for looking up hypervectors
        seed: Random seed for role vector generation

    Example:
        >>> from vsar.kernel.vsa_backend import FHRRBackend
        >>> from vsar.symbols.registry import SymbolRegistry
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> registry = SymbolRegistry(dim=backend.dim, seed=42)
        >>> encoder = RoleFillerEncoder(backend, registry, seed=42)
        >>>
        >>> # Encode ground atom: parent(alice, bob)
        >>> atom_vec = encoder.encode_atom("parent", ["alice", "bob"])
        >>>
        >>> # Encode query: parent(alice, X)
        >>> query_vec = encoder.encode_query("parent", ["alice", None])
    """

    def __init__(self, backend: KernelBackend, registry: SymbolRegistry, seed: int = 42):
        super().__init__(backend, registry, seed)

    def _get_role_vector(self, position: int) -> jnp.ndarray:
        """
        Get role vector for a given argument position.

        Args:
            position: 1-indexed argument position

        Returns:
            Role vector for the position
        """
        role_name = f"ARG{position}"
        return self.registry.register(SymbolSpace.ARG_ROLES, role_name)

    def encode_atom(self, predicate: str, args: list[str]) -> jnp.ndarray:
        """
        Encode a ground atom using hybrid predicate-bound + shift encoding.

        Formula: enc(p(t1,...,tk)) = P_p ⊗ (shift(t1,1) ⊕ shift(t2,2) ⊕ ... ⊕ shift(tk,k))

        Where:
        - P_p is the predicate vector from PREDICATES space
        - shift(ti, i) is circular permutation of entity ti by position i
        - Binding predicate allows distinguishing different predicates
        - Shift encoding provides clean invertible positional decoding

        Args:
            predicate: Predicate name (e.g., "parent") - encoded into vector
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

        # Get predicate vector
        pred_vec = self.registry.register(SymbolSpace.PREDICATES, predicate)

        # Encode each argument with shift-based positional encoding
        shifted_args = []
        for i, arg in enumerate(args):
            position = i + 1  # 1-indexed positions

            # Get entity vector
            entity_vec = self.registry.register(SymbolSpace.ENTITIES, arg)

            # Shift entity by position: shift(entity, position)
            shifted = self.backend.permute(entity_vec, position)
            shifted_args.append(shifted)

        # Bundle all shifted arguments using plain sum (NOT backend.bundle which may add noise)
        # CRITICAL: Use plain addition to preserve exact linear superposition for interference cancellation
        args_bundle = sum(shifted_args)  # Plain element-wise sum

        # Bind predicate with arguments bundle: P_p ⊗ args_bundle
        atom_vec = self.backend.bind(pred_vec, args_bundle)

        return self.backend.normalize(atom_vec)

    def encode_query(self, predicate: str, args: list[str | None]) -> jnp.ndarray:
        """
        Encode a query pattern using hybrid shift-based encoding.

        Query patterns have variables represented as None. Only bound
        arguments are included in the encoding using shift-based positional encoding.

        Formula: enc(p(t1,X,...)) = P_p ⊗ (shift(enc(t1), 1) ⊕ ...)
        (only bound arguments included)

        Args:
            predicate: Predicate name (e.g., "parent") - encoded into vector
            args: List of entity names or None for variables
                  (e.g., ["alice", None] for parent(alice, X))

        Returns:
            Hypervector encoding of the query pattern (predicate + bound args)

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

        # Get predicate vector
        pred_vec = self.registry.register(SymbolSpace.PREDICATES, predicate)

        # Encode only bound arguments with shift-based positional encoding
        shifted_args = []
        for i, arg in enumerate(args):
            if arg is not None:  # Skip variables
                position = i + 1  # 1-indexed positions

                # Get entity vector
                entity_vec = self.registry.register(SymbolSpace.ENTITIES, arg)

                # Shift entity by position: shift(entity, position)
                shifted = self.backend.permute(entity_vec, position)
                shifted_args.append(shifted)

        # Bundle all shifted arguments using plain sum for exact linear superposition
        args_bundle = sum(shifted_args)  # Plain element-wise sum

        # Bind predicate with arguments bundle
        query_vec = self.backend.bind(pred_vec, args_bundle)

        return self.backend.normalize(query_vec)
