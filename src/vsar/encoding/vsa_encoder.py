"""VSA encoder using role-filler binding."""

import jax.numpy as jnp

from vsar.encoding.base import AtomEncoder
from vsar.encoding.roles import RoleVectorManager
from vsar.kernel.base import KernelBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class VSAEncoder(AtomEncoder):
    """
    VSA encoder using role-filler binding.

    Encodes atoms using the formula:
        enc(p(t1,...,tk)) = hv(p) ⊗ ((hv(ρ1) ⊗ hv(t1)) ⊕ ... ⊕ (hv(ρk) ⊗ hv(tk)))

    Where:
    - hv(p) is the predicate hypervector from RELATIONS space
    - hv(ti) is the entity hypervector from ENTITIES space
    - hv(ρi) is the role vector for position i
    - ⊗ is the bind operation (circular convolution for FHRR)
    - ⊕ is the bundle operation (element-wise sum + normalization)

    Args:
        backend: Kernel backend for hypervector operations
        registry: Symbol registry for looking up hypervectors
        seed: Random seed for deterministic role vector generation

    Example:
        >>> from vsar.kernel.vsa_backend import FHRRBackend
        >>> from vsar.symbols.registry import SymbolRegistry
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> registry = SymbolRegistry(backend, seed=42)
        >>> encoder = VSAEncoder(backend, registry, seed=42)
        >>>
        >>> # Register symbols
        >>> registry.register(SymbolSpace.RELATIONS, "parent")
        >>> registry.register(SymbolSpace.ENTITIES, "alice")
        >>> registry.register(SymbolSpace.ENTITIES, "bob")
        >>>
        >>> # Encode ground atom: parent(alice, bob)
        >>> atom_vec = encoder.encode_atom("parent", ["alice", "bob"])
        >>>
        >>> # Encode query: parent(alice, X)
        >>> query_vec = encoder.encode_query("parent", ["alice", None])
    """

    def __init__(
        self, backend: KernelBackend, registry: SymbolRegistry, seed: int = 42
    ):
        super().__init__(backend, registry, seed)
        self.role_manager = RoleVectorManager(backend, seed)

    def encode_atom(self, predicate: str, args: list[str]) -> jnp.ndarray:
        """
        Encode a ground atom into a hypervector.

        Uses role-filler binding to encode the predicate and its arguments.

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
        if not args:
            raise ValueError("Arguments list cannot be empty")

        # Get predicate hypervector from RELATIONS space
        pred_vec = self.registry.register(SymbolSpace.RELATIONS, predicate)

        # Encode each argument with its role
        role_filler_pairs = []
        for i, arg in enumerate(args):
            position = i + 1  # 1-indexed positions
            role_vec = self.role_manager.get_role(position)
            entity_vec = self.registry.register(SymbolSpace.ENTITIES, arg)

            # Bind role to filler: ρi ⊗ ti
            role_filler = self.backend.bind(role_vec, entity_vec)
            role_filler_pairs.append(role_filler)

        # Bundle all role-filler pairs: (ρ1 ⊗ t1) ⊕ (ρ2 ⊗ t2) ⊕ ...
        bundled = self.backend.bundle(role_filler_pairs)

        # Bind predicate to the bundle: hv(p) ⊗ bundle
        atom_vec = self.backend.bind(pred_vec, bundled)

        return self.backend.normalize(atom_vec)

    def encode_query(
        self, predicate: str, args: list[str | None]
    ) -> jnp.ndarray:
        """
        Encode a query pattern into a hypervector.

        Query patterns have variables represented as None. Only bound
        arguments are included in the encoding.

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
        if not args:
            raise ValueError("Arguments list cannot be empty")

        # Check that at least one argument is bound
        if all(arg is None for arg in args):
            raise ValueError("At least one argument must be bound (not None)")

        # Get predicate hypervector from RELATIONS space
        pred_vec = self.registry.register(SymbolSpace.RELATIONS, predicate)

        # Encode only bound arguments with their roles
        role_filler_pairs = []
        for i, arg in enumerate(args):
            if arg is not None:  # Skip variables
                position = i + 1  # 1-indexed positions
                role_vec = self.role_manager.get_role(position)
                entity_vec = self.registry.register(SymbolSpace.ENTITIES, arg)

                # Bind role to filler: ρi ⊗ ti
                role_filler = self.backend.bind(role_vec, entity_vec)
                role_filler_pairs.append(role_filler)

        # Bundle all role-filler pairs
        bundled = self.backend.bundle(role_filler_pairs)

        # Bind predicate to the bundle
        query_vec = self.backend.bind(pred_vec, bundled)

        return self.backend.normalize(query_vec)
