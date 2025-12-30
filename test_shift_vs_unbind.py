"""
Compare shift-based encoding vs bind/unbind encoding for query retrieval.

This test compares two approaches:
1. Current: Shift-based positional encoding with resonator filtering
2. New: Role-filler binding with unbind-based retrieval

Goal: Determine which approach gives better query results.
"""

import jax
import jax.numpy as jnp
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class ShiftBasedApproach:
    """Current approach: shift-based encoding + resonator filtering."""

    def __init__(self, backend, registry):
        self.backend = backend
        self.registry = registry

    def encode_fact(self, predicate, args):
        """Encode fact using shift: shift(arg1, 1) + shift(arg2, 2) + ..."""
        # Get entity vectors
        entity_vecs = [
            self.registry.register(SymbolSpace.ENTITIES, arg) for arg in args
        ]

        # Shift each entity by its position (1-indexed)
        shifted_vecs = [
            self.backend.permute(vec, pos + 1)
            for pos, vec in enumerate(entity_vecs)
        ]

        # Bundle (sum and normalize)
        fact_vec = self.backend.bundle(shifted_vecs)
        return fact_vec

    def query(self, fact_vecs, bound_pos, bound_entity, var_pos, k=10):
        """
        Query using resonator filtering.

        Args:
            fact_vecs: List of encoded fact vectors
            bound_pos: Position of bound argument (1-indexed)
            bound_entity: Entity name at bound position
            var_pos: Position of variable to retrieve (1-indexed)
            k: Number of results

        Returns:
            List of (entity, similarity) tuples
        """
        # Get bound entity vector
        bound_vec = self.registry.register(SymbolSpace.ENTITIES, bound_entity)

        # Resonator filtering: weight facts by how well they match
        weights = []
        for fact_vec in fact_vecs:
            # Decode bound position
            decoded = self.backend.permute(fact_vec, -bound_pos)
            # Compute similarity
            sim = self.backend.similarity(decoded, bound_vec)
            weights.append(max(0.0, float(sim)))

        # Create weighted bundle
        if not any(w > 0 for w in weights):
            return []

        weighted_bundle = jnp.zeros_like(fact_vecs[0])
        for i, fact_vec in enumerate(fact_vecs):
            weighted_bundle = weighted_bundle + weights[i] * fact_vec
        weighted_bundle = self.backend.normalize(weighted_bundle)

        # Decode variable position
        entity_vec = self.backend.permute(weighted_bundle, -var_pos)

        # Cleanup: find top-k entities
        from vsar.retrieval.cleanup import cleanup

        results = cleanup(SymbolSpace.ENTITIES, entity_vec, self.registry, self.backend, k)
        return results


class BindUnbindApproach:
    """New approach: role-filler binding with unbind-based retrieval."""

    def __init__(self, backend, registry):
        self.backend = backend
        self.registry = registry
        # Create role vectors for each argument position
        self.role_vecs = {}

    def get_role_vec(self, position):
        """Get or create role vector for a position."""
        if position not in self.role_vecs:
            key = jax.random.PRNGKey(1000 + position)
            self.role_vecs[position] = self.backend.generate_random(key, (self.backend.dimension,))
        return self.role_vecs[position]

    def encode_fact(self, predicate, args):
        """Encode fact using bind: bind(role1, arg1) + bind(role2, arg2) + ..."""
        # Get entity vectors
        entity_vecs = [
            self.registry.register(SymbolSpace.ENTITIES, arg) for arg in args
        ]

        # Bind each entity with its role vector
        bound_vecs = []
        for pos, vec in enumerate(entity_vecs):
            role = self.get_role_vec(pos + 1)  # 1-indexed
            bound = self.backend.bind(role, vec)
            bound_vecs.append(bound)

        # Bundle (sum and normalize)
        fact_vec = self.backend.bundle(bound_vecs)
        return fact_vec

    def query(self, fact_vecs, bound_pos, bound_entity, var_pos, k=10):
        """
        Query using unbind-based retrieval.

        Args:
            fact_vecs: List of encoded fact vectors
            bound_pos: Position of bound argument (1-indexed)
            bound_entity: Entity name at bound position
            var_pos: Position of variable to retrieve (1-indexed)
            k: Number of results

        Returns:
            List of (entity, similarity) tuples
        """
        # Get bound entity vector and role
        bound_vec = self.registry.register(SymbolSpace.ENTITIES, bound_entity)
        bound_role = self.get_role_vec(bound_pos)

        # Create query probe: bind(role, entity)
        query_probe = self.backend.bind(bound_role, bound_vec)

        # Filter facts by similarity to probe
        weights = []
        for fact_vec in fact_vecs:
            sim = self.backend.similarity(fact_vec, query_probe)
            weights.append(max(0.0, float(sim)))

        # Create weighted bundle
        if not any(w > 0 for w in weights):
            return []

        weighted_bundle = jnp.zeros_like(fact_vecs[0])
        for i, fact_vec in enumerate(fact_vecs):
            weighted_bundle = weighted_bundle + weights[i] * fact_vec
        weighted_bundle = self.backend.normalize(weighted_bundle)

        # Unbind variable position
        var_role = self.get_role_vec(var_pos)
        entity_vec = self.backend.unbind(weighted_bundle, var_role)

        # Cleanup: find top-k entities
        from vsar.retrieval.cleanup import cleanup

        results = cleanup(SymbolSpace.ENTITIES, entity_vec, self.registry, self.backend, k)
        return results


def run_comparison():
    """Run comparison test on parent(alice, X) query."""
    print("\n" + "=" * 60)
    print("SHIFT-BASED vs BIND/UNBIND COMPARISON")
    print("=" * 60)

    # Setup
    backend = FHRRBackend(dim=512, seed=42)
    registry = SymbolRegistry(backend, seed=42)

    shift_approach = ShiftBasedApproach(backend, registry)
    bind_approach = BindUnbindApproach(backend, registry)

    # Test data: parent facts
    facts = [
        ("alice", "bob"),
        ("alice", "carol"),
        ("bob", "dave"),
        ("carol", "eve"),
    ]

    print("\nTest Data (parent facts):")
    for parent, child in facts:
        print(f"  parent({parent}, {child})")

    print("\nQuery: parent(alice, X)?")
    print("Expected: bob and carol with high scores")
    print()

    # Encode facts with both approaches
    print("Encoding facts...")
    shift_fact_vecs = [shift_approach.encode_fact("parent", [p, c]) for p, c in facts]
    bind_fact_vecs = [bind_approach.encode_fact("parent", [p, c]) for p, c in facts]

    # Query with both approaches
    print("\n" + "-" * 60)
    print("SHIFT-BASED APPROACH")
    print("-" * 60)
    shift_results = shift_approach.query(
        shift_fact_vecs,
        bound_pos=1,
        bound_entity="alice",
        var_pos=2,
        k=10
    )

    print("Results:")
    for i, (entity, score) in enumerate(shift_results[:5], 1):
        marker = "  <-- EXPECTED" if entity in ["bob", "carol"] else ""
        print(f"  {i}. {entity:10s} : {score:.4f}{marker}")

    print("\n" + "-" * 60)
    print("BIND/UNBIND APPROACH")
    print("-" * 60)
    bind_results = bind_approach.query(
        bind_fact_vecs,
        bound_pos=1,
        bound_entity="alice",
        var_pos=2,
        k=10
    )

    print("Results:")
    for i, (entity, score) in enumerate(bind_results[:5], 1):
        marker = "  <-- EXPECTED" if entity in ["bob", "carol"] else ""
        print(f"  {i}. {entity:10s} : {score:.4f}{marker}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    def analyze_results(results, approach_name):
        top5 = results[:5]
        expected = {"bob", "carol"}
        found = {entity for entity, _ in top5}

        # Check if expected results are in top 2
        top2 = {entity for entity, _ in results[:2]}
        top2_match = expected == top2

        # Average score of expected results
        expected_scores = [score for entity, score in results if entity in expected]
        avg_expected = sum(expected_scores) / len(expected_scores) if expected_scores else 0

        # Average score of top 2
        top2_scores = [score for _, score in results[:2]]
        avg_top2 = sum(top2_scores) / len(top2_scores) if top2_scores else 0

        print(f"\n{approach_name}:")
        print(f"  Top 2 correct: {top2_match}")
        print(f"  Expected in top 5: {expected.intersection(found)}")
        print(f"  Avg score (expected): {avg_expected:.4f}")
        print(f"  Avg score (top 2): {avg_top2:.4f}")

        return {
            "top2_correct": top2_match,
            "avg_expected": avg_expected,
            "avg_top2": avg_top2,
        }

    shift_analysis = analyze_results(shift_results, "Shift-based")
    bind_analysis = analyze_results(bind_results, "Bind/unbind")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if shift_analysis["avg_expected"] > bind_analysis["avg_expected"]:
        print("\nWINNER: Shift-based approach")
        print(f"  Better average score for expected results:")
        print(f"    Shift: {shift_analysis['avg_expected']:.4f}")
        print(f"    Bind:  {bind_analysis['avg_expected']:.4f}")
        print("\nRECOMMENDATION: Keep current shift-based approach")
    elif bind_analysis["avg_expected"] > shift_analysis["avg_expected"]:
        print("\nWINNER: Bind/unbind approach")
        print(f"  Better average score for expected results:")
        print(f"    Bind:  {bind_analysis['avg_expected']:.4f}")
        print(f"    Shift: {shift_analysis['avg_expected']:.4f}")
        print("\nRECOMMENDATION: Consider switching to bind/unbind approach")
    else:
        print("\nTIE: Both approaches perform similarly")
        print(f"  Average scores: {shift_analysis['avg_expected']:.4f}")
        print("\nRECOMMENDATION: Keep current shift-based approach (simpler)")

    print()


if __name__ == "__main__":
    run_comparison()
