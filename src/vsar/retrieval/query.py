"""Query retrieval interface."""

from typing import Any

import jax.numpy as jnp

from vsar.encoding.roles import RoleVectorManager
from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.base import KernelBackend
from vsar.retrieval.cleanup import cleanup
from vsar.retrieval.unbind import extract_variable_binding
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class Retriever:
    """
    Top-k retrieval for VSAR queries.

    Orchestrates the retrieval pipeline:
    1. Encode query pattern with bound arguments
    2. Get KB bundle for predicate
    3. Unbind query to extract variable binding
    4. Cleanup to find top-k matching symbols

    Args:
        backend: Kernel backend
        registry: Symbol registry
        kb: Knowledge base
        encoder: Atom encoder
        role_manager: Role vector manager

    Example:
        >>> from vsar.kernel.vsa_backend import FHRRBackend
        >>> from vsar.symbols.registry import SymbolRegistry
        >>> from vsar.kb.store import KnowledgeBase
        >>> from vsar.encoding.vsa_encoder import VSAEncoder
        >>> from vsar.encoding.roles import RoleVectorManager
        >>>
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> registry = SymbolRegistry(backend, seed=42)
        >>> kb = KnowledgeBase(backend)
        >>> encoder = VSAEncoder(backend, registry, seed=42)
        >>> role_manager = RoleVectorManager(backend, seed=42)
        >>>
        >>> retriever = Retriever(backend, registry, kb, encoder, role_manager)
        >>>
        >>> # Insert facts
        >>> atom_vec = encoder.encode_atom("parent", ["alice", "bob"])
        >>> kb.insert("parent", atom_vec, ("alice", "bob"))
        >>>
        >>> # Query: parent(alice, X)
        >>> results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)
        >>> results[0]
        ('bob', 0.85)
    """

    def __init__(
        self,
        backend: KernelBackend,
        registry: SymbolRegistry,
        kb: KnowledgeBase,
        encoder: VSAEncoder,
        role_manager: RoleVectorManager,
    ):
        self.backend = backend
        self.registry = registry
        self.kb = kb
        self.encoder = encoder
        self.role_manager = role_manager

    def retrieve(
        self,
        predicate: str,
        var_position: int,
        bound_args: dict[str, str],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Retrieve top-k bindings for a variable in a query.

        Args:
            predicate: Predicate name (e.g., "parent")
            var_position: Position of the variable (1-indexed)
            bound_args: Dictionary mapping position (as string) to entity name
                       e.g., {"1": "alice"} for parent(alice, X)
            k: Number of top results to return

        Returns:
            List of (entity_name, similarity_score) tuples, sorted by score descending

        Raises:
            ValueError: If predicate not in KB
            ValueError: If var_position is in bound_args

        Example:
            >>> # Query: parent(alice, X) where X is at position 2
            >>> results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)
            >>> results[0]
            ('bob', 0.85)
        """
        # Validate inputs
        if not self.kb.has_predicate(predicate):
            raise ValueError(f"Predicate '{predicate}' not found in KB")

        if str(var_position) in bound_args:
            raise ValueError(
                f"Variable position {var_position} cannot be in bound_args"
            )

        # Get KB bundle for predicate
        kb_bundle = self.kb.get_bundle(predicate)
        if kb_bundle is None:
            return []

        # Determine arity from KB facts
        facts = self.kb.get_facts(predicate)
        if not facts:
            return []

        arity = len(facts[0])

        # Build argument list with None for variable position
        args: list[str | None] = [None] * arity
        for pos_str, entity in bound_args.items():
            pos = int(pos_str)
            if pos < 1 or pos > arity:
                raise ValueError(f"Position {pos} out of range for arity {arity}")
            args[pos - 1] = entity

        # Encode query pattern
        query_vec = self.encoder.encode_query(predicate, args)

        # Get role vector for variable position
        role_vec = self.role_manager.get_role(var_position)

        # Extract variable binding
        entity_vec = extract_variable_binding(
            kb_bundle, query_vec, role_vec, self.backend
        )

        # Cleanup to find top-k matches
        results = cleanup(
            SymbolSpace.ENTITIES, entity_vec, self.registry, self.backend, k
        )

        return results

    def retrieve_all_vars(
        self,
        predicate: str,
        bound_args: dict[str, str],
        k: int = 10,
    ) -> dict[int, list[tuple[str, float]]]:
        """
        Retrieve bindings for all unbound positions.

        Args:
            predicate: Predicate name
            bound_args: Dictionary mapping position to entity name
            k: Number of top results per variable

        Returns:
            Dictionary mapping variable position to top-k results

        Example:
            >>> # Query: parent(alice, X, Y) - retrieve both X and Y
            >>> results = retriever.retrieve_all_vars(
            ...     "grandparent", {"1": "alice"}, k=5
            ... )
            >>> results[2]  # Results for position 2
            [('bob', 0.85), ('carol', 0.72)]
            >>> results[3]  # Results for position 3
            [('dave', 0.78), ('eve', 0.65)]
        """
        # Get arity from KB facts
        facts = self.kb.get_facts(predicate)
        if not facts:
            return {}

        arity = len(facts[0])

        # Find unbound positions
        bound_positions = {int(pos) for pos in bound_args.keys()}
        unbound_positions = [
            pos for pos in range(1, arity + 1) if pos not in bound_positions
        ]

        # Retrieve for each unbound position
        results = {}
        for var_pos in unbound_positions:
            results[var_pos] = self.retrieve(predicate, var_pos, bound_args, k)

        return results
