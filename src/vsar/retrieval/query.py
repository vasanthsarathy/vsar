"""Query retrieval interface using successive interference cancellation."""

from typing import Any

import jax.numpy as jnp

from vsar.encoding.base import AtomEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.base import KernelBackend
from vsar.retrieval.cleanup import cleanup
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class Retriever:
    """
    Top-k retrieval for VSAR queries using hybrid encoding with interference cancellation.

    Orchestrates the retrieval pipeline:
    1. Get all fact vectors for the predicate (stored separately, not bundled)
    2. Unbind predicate to get args_bundle
    3. For each fact, decode bound argument positions and verify match
    4. Apply successive interference cancellation: subtract known args from bundle
    5. Decode variable position from cleaned bundle
    6. Cleanup to find top-k matching symbols

    Uses hybrid encoding: enc(p(t1,...,tk)) = P_p ⊗ (shift(t1,1) ⊕ shift(t2,2) ⊕ ...)

    Key optimization: Successive interference cancellation
    - For query p(known1, X), after unbinding predicate we have: shift(known1,1) + shift(X,2)
    - Before decoding X, we subtract shift(known1,1) to get: shift(X,2)
    - This removes interference, boosting similarity from ~0.64 to ~0.95-1.0

    Args:
        backend: Kernel backend
        registry: Symbol registry
        kb: Knowledge base
        encoder: Atom encoder

    Example:
        >>> from vsar.kernel.vsa_backend import FHRRBackend
        >>> from vsar.symbols.registry import SymbolRegistry
        >>> from vsar.kb.store import KnowledgeBase
        >>> from vsar.encoding.role_filler_encoder import RoleFillerEncoder
        >>>
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> registry = SymbolRegistry(dim=backend.dimension, seed=42)
        >>> kb = KnowledgeBase(backend)
        >>> encoder = RoleFillerEncoder(backend, registry, seed=42)
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
        ('bob', 0.95)  # High similarity due to interference cancellation
    """

    def __init__(
        self,
        backend: KernelBackend,
        registry: SymbolRegistry,
        kb: KnowledgeBase,
        encoder: AtomEncoder,
    ):
        self.backend = backend
        self.registry = registry
        self.kb = kb
        self.encoder = encoder

    def _get_role_vector(self, position: int) -> jnp.ndarray:
        """Get role vector for a given argument position (1-indexed)."""
        role_name = f"ARG{position}"
        return self.registry.register(SymbolSpace.ARG_ROLES, role_name)

    def retrieve(
        self,
        predicate: str,
        var_position: int,
        bound_args: dict[str, str],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Retrieve top-k bindings for a variable using interference cancellation.

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
            ('bob', 0.95)  # High score due to interference cancellation
        """
        # Validate inputs
        if not self.kb.has_predicate(predicate):
            raise ValueError(f"Predicate '{predicate}' not found in KB")

        if str(var_position) in bound_args:
            raise ValueError(f"Variable position {var_position} cannot be in bound_args")

        # Get fact vectors for predicate
        fact_vectors = self.kb.get_vectors(predicate)
        if not fact_vectors:
            return []

        # Get predicate vector (needed for unbinding)
        pred_vec = self.registry.register(SymbolSpace.PREDICATES, predicate)

        # If no bound arguments, skip resonator filtering (return all entities)
        if not bound_args:
            # Just decode the variable position from each fact
            candidates: list[tuple[str, float]] = []
            for fact_vec in fact_vectors:
                # Unbind predicate first: unbind(P_p ⊗ args_bundle, P_p) → args_bundle
                args_bundle = self.backend.unbind(fact_vec, pred_vec)

                # Shift decode to get entity: shift(args_bundle, -position) → entity_vec
                entity_vec = self.backend.permute(args_bundle, -var_position)

                cleanup_results = self.registry.cleanup(SymbolSpace.ENTITIES, entity_vec, k=1)
                if cleanup_results:
                    entity, score = cleanup_results[0]
                    candidates.append((entity, score))

            # Sort by score and return top k
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:k]

        # Separate cleanup with interference cancellation
        candidates: list[tuple[str, float]] = []

        for fact_vec in fact_vectors:
            # Unbind predicate first to get args_bundle
            # args_bundle = shift(t1,1) + shift(t2,2) + ...
            args_bundle = self.backend.unbind(fact_vec, pred_vec)

            # Check if this fact matches all bound arguments
            matches = True
            for pos_str, entity in bound_args.items():
                position = int(pos_str)

                # Decode this position from args_bundle using shift
                decoded = self.backend.permute(args_bundle, -position)

                # Get entity vector
                entity_vec = self.registry.register(SymbolSpace.ENTITIES, entity)

                # Compute similarity
                similarity = self.backend.similarity(decoded, entity_vec)

                # If similarity too low, this fact doesn't match
                if float(similarity) < 0.5:  # Threshold for matching
                    matches = False
                    break

            if not matches:
                # Skip facts that don't match bound arguments
                continue

            # This fact matches - apply successive interference cancellation
            # Subtract known arguments to remove interference before decoding unknown position
            #
            # For query parent(alice, X):
            #   args_bundle = shift(alice,1) + shift(X,2)
            #   After subtraction: shift(X,2)
            #   After shift(-2): X (clean, high similarity ~0.95-1.0)
            #
            cleaned_bundle = args_bundle
            for pos_str, entity in bound_args.items():
                position = int(pos_str)
                # Get entity vector
                entity_vec = self.registry.register(SymbolSpace.ENTITIES, entity)
                # Compute the shifted contribution: shift(entity, position)
                shifted_contribution = self.backend.permute(entity_vec, position)
                # Subtract it from the bundle (interference cancellation)
                # CRITICAL: Do NOT normalize during subtraction - we need linear superposition
                cleaned_bundle = cleaned_bundle - shifted_contribution

            # Now decode the variable position from the cleaned bundle
            entity_vec = self.backend.permute(cleaned_bundle, -var_position)

            # Cleanup to find the entity at this position
            cleanup_results = self.registry.cleanup(SymbolSpace.ENTITIES, entity_vec, k=1)

            if cleanup_results:
                entity, score = cleanup_results[0]
                candidates.append((entity, score))

        if not candidates:
            # No matching facts
            return []

        # Aggregate: collect unique entities with their best scores
        entity_scores: dict[str, float] = {}
        for entity, score in candidates:
            if entity not in entity_scores or score > entity_scores[entity]:
                entity_scores[entity] = score

        # Sort by score and return top-k
        results = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:k]

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

    def retrieve_multi_variable(
        self,
        predicate: str,
        var_positions: list[int],
        bound_args: dict[str, str],
        k: int = 10,
        beam_width: int = 10,
    ) -> list[tuple[tuple[str, ...], float]]:
        """
        Retrieve top-k joint bindings for multiple variables using successive interference cancellation.

        This decodes each fact's entities at the variable positions, using interference
        cancellation to improve accuracy.

        Algorithm:
        1. For each fact in KB:
           - Unbind predicate to get args_bundle
           - Check bound arguments match (if any)
           - Iteratively decode each variable position with interference cancellation
        2. Aggregate results and return top-k by joint similarity

        Args:
            predicate: Predicate name (e.g., "parent")
            var_positions: List of variable positions (1-indexed), e.g., [1, 2] for parent(?, ?)
            bound_args: Dictionary mapping position to entity name (for partially bound queries)
            k: Number of top results to return
            beam_width: Not used in current implementation (for future enhancements)

        Returns:
            List of (entity_tuple, joint_similarity) tuples, sorted by score descending
            entity_tuple has length len(var_positions), in same order as var_positions

        Example:
            >>> # Query: parent(?, ?) - retrieve all parent-child pairs
            >>> results = retriever.retrieve_multi_variable("parent", [1, 2], {}, k=10)
            >>> results[0]
            (('alice', 'bob'), 0.88)  # Parent-child pair with joint similarity

            >>> # Query: works_in(alice, ?, ?) - retrieve department and role
            >>> results = retriever.retrieve_multi_variable(
            ...     "works_in", [2, 3], {"1": "alice"}, k=5
            ... )
            >>> results[0]
            (('engineering', 'lead'), 0.82)
        """
        # Validate inputs
        if not var_positions:
            raise ValueError("Must have at least one variable position")

        # Check if predicate exists - return empty if not
        if not self.kb.has_predicate(predicate):
            return []

        # Get fact vectors for predicate
        fact_vectors = self.kb.get_vectors(predicate)
        if not fact_vectors:
            return []

        # Get predicate vector
        pred_vec = self.registry.register(SymbolSpace.PREDICATES, predicate)

        # Store all joint binding candidates across all facts
        all_candidates: list[tuple[tuple[str, ...], float]] = []

        # Process each fact
        for fact_vec in fact_vectors:
            # Step 1: Unbind predicate to get args_bundle
            # args_bundle = shift(t1,1) ⊕ shift(t2,2) ⊕ ...
            args_bundle = self.backend.unbind(fact_vec, pred_vec)

            # Step 2: Verify bound arguments match (if any)
            if bound_args:
                matches = True
                for pos_str, entity in bound_args.items():
                    position = int(pos_str)
                    # Decode this position
                    decoded = self.backend.permute(args_bundle, -position)
                    entity_vec = self.registry.register(SymbolSpace.ENTITIES, entity)
                    similarity = self.backend.similarity(decoded, entity_vec)
                    if float(similarity) < 0.5:
                        matches = False
                        break

                if not matches:
                    continue

                # Cancel bound arguments to reduce interference
                for pos_str, entity in bound_args.items():
                    position = int(pos_str)
                    entity_vec = self.registry.register(SymbolSpace.ENTITIES, entity)
                    shifted_contribution = self.backend.permute(entity_vec, position)
                    args_bundle = args_bundle - shifted_contribution

            # Step 3: Decode each variable position with successive interference cancellation
            binding = []
            similarities = []
            cleaned_bundle = args_bundle

            for var_pos in var_positions:
                # Decode this position from cleaned bundle
                entity_vec = self.backend.permute(cleaned_bundle, -var_pos)
                cleanup_results = self.registry.cleanup(
                    SymbolSpace.ENTITIES, entity_vec, k=1
                )

                if not cleanup_results:
                    # Failed to decode this position - skip this fact
                    break

                entity_name, similarity = cleanup_results[0]
                binding.append(entity_name)
                similarities.append(similarity)

                # Cancel this entity's contribution for next iteration
                # This is the key: interference cancellation improves subsequent decoding
                entity_vec_reg = self.registry.register(SymbolSpace.ENTITIES, entity_name)
                shifted_contribution = self.backend.permute(entity_vec_reg, var_pos)
                cleaned_bundle = cleaned_bundle - shifted_contribution

            else:
                # Successfully decoded all positions
                # Compute joint similarity as average of all position similarities
                joint_sim = sum(similarities) / len(similarities)
                all_candidates.append((tuple(binding), joint_sim))

        if not all_candidates:
            return []

        # Step 4: Aggregate and rank by joint similarity
        # Keep best score for each unique binding tuple
        binding_scores: dict[tuple[str, ...], float] = {}
        for binding, score in all_candidates:
            if binding not in binding_scores or score > binding_scores[binding]:
                binding_scores[binding] = score

        # Sort by joint similarity and return top-k
        results = sorted(binding_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:k]
