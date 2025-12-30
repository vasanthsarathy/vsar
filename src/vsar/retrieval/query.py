"""Query retrieval interface using resonator filtering."""

from typing import Any

import jax.numpy as jnp

from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.base import KernelBackend
from vsar.retrieval.cleanup import cleanup
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class Retriever:
    """
    Top-k retrieval for VSAR queries using resonator filtering.

    Orchestrates the retrieval pipeline using shift-based encoding:
    1. Get all fact vectors for the predicate (stored separately, not bundled)
    2. For each fact, decode bound argument positions and compute similarity
    3. Weight facts by how well they match bound arguments (resonator filtering)
    4. Create weighted bundle of matching facts
    5. Decode variable position from weighted bundle
    6. Cleanup to find top-k matching symbols

    This approach avoids bind/unbind operations which are broken in vsax.

    Args:
        backend: Kernel backend
        registry: Symbol registry
        kb: Knowledge base
        encoder: Atom encoder

    Example:
        >>> from vsar.kernel.vsa_backend import FHRRBackend
        >>> from vsar.symbols.registry import SymbolRegistry
        >>> from vsar.kb.store import KnowledgeBase
        >>> from vsar.encoding.vsa_encoder import VSAEncoder
        >>>
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> registry = SymbolRegistry(backend, seed=42)
        >>> kb = KnowledgeBase(backend)
        >>> encoder = VSAEncoder(backend, registry, seed=42)
        >>>
        >>> retriever = Retriever(backend, registry, kb, encoder)
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
    ):
        self.backend = backend
        self.registry = registry
        self.kb = kb
        self.encoder = encoder

    def retrieve(
        self,
        predicate: str,
        var_position: int,
        bound_args: dict[str, str],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Retrieve top-k bindings for a variable in a query using resonator filtering.

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
            ValueError: If no bound arguments provided

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
            raise ValueError(f"Variable position {var_position} cannot be in bound_args")

        if not bound_args:
            raise ValueError("At least one argument must be bound for querying")

        # Get fact vectors for predicate
        fact_vectors = self.kb.get_vectors(predicate)
        if not fact_vectors:
            return []

        # Resonator filtering: compute weights for each fact
        weights = []
        for fact_vec in fact_vectors:
            # For each bound argument, check if this fact matches
            fact_weight = 1.0
            for pos_str, entity in bound_args.items():
                position = int(pos_str)

                # Decode this position from the fact
                decoded = self.backend.permute(fact_vec, -position)

                # Get entity vector
                entity_vec = self.registry.register(SymbolSpace.ENTITIES, entity)

                # Compute similarity
                similarity = self.backend.similarity(decoded, entity_vec)

                # Multiply weights (all bound args must match)
                fact_weight *= max(0.0, float(similarity))

            weights.append(fact_weight)

        # Create weighted bundle
        if not any(w > 0 for w in weights):
            # No matching facts
            return []

        # Weighted sum of fact vectors
        weighted_bundle = jnp.zeros_like(fact_vectors[0])
        for i, fact_vec in enumerate(fact_vectors):
            weighted_bundle = weighted_bundle + weights[i] * fact_vec

        # Normalize
        weighted_bundle = self.backend.normalize(weighted_bundle)

        # Decode variable position
        entity_vec = self.backend.permute(weighted_bundle, -var_position)

        # Cleanup to find top-k matches
        results = cleanup(SymbolSpace.ENTITIES, entity_vec, self.registry, self.backend, k)

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
        unbound_positions = [pos for pos in range(1, arity + 1) if pos not in bound_positions]

        # Retrieve for each unbound position
        results = {}
        for var_pos in unbound_positions:
            results[var_pos] = self.retrieve(predicate, var_pos, bound_args, k)

        return results
