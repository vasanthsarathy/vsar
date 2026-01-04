"""Sanity checks for VSA encoding as recommended by expert."""

import jax.numpy as jnp

from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


def test_bind_unbind_identity():
    """Test if bind/unbind is truly inverse (should be ~0.95-1.0)."""
    print("\n" + "="*80)
    print("SANITY CHECK 1: Bind/Unbind Identity")
    print("="*80)

    backend = FHRRBackend(dim=8192, seed=42)
    registry = SymbolRegistry(dim=8192, seed=42)

    # Get random entity and role vectors
    entity_a = registry.register(SymbolSpace.ENTITIES, "alice")
    role_1 = registry.register(SymbolSpace.ARG_ROLES, "ARG1")

    # Test: sim(unbind(bind(r, a), r), a) should be ~1.0
    bound = backend.bind(role_1, entity_a)
    unbound = backend.unbind(bound, role_1)

    similarity = backend.similarity(unbound, entity_a)

    print(f"Entity: alice")
    print(f"Role: ARG1")
    print(f"Similarity after bind->unbind: {float(similarity):.6f}")

    if similarity > 0.95:
        print("PASS PASS: Bind/unbind is invertible")
    else:
        print("FAIL FAIL: Bind/unbind is NOT properly invertible!")
        print("   This indicates backend implementation issue")

    return float(similarity)


def test_bundling_normalization():
    """Check how bundling normalizes vectors."""
    print("\n" + "="*80)
    print("SANITY CHECK 2: Bundling Normalization")
    print("="*80)

    backend = FHRRBackend(dim=8192, seed=42)
    registry = SymbolRegistry(dim=8192, seed=42)

    # Get two entity vectors
    entity_a = registry.register(SymbolSpace.ENTITIES, "alice")
    entity_b = registry.register(SymbolSpace.ENTITIES, "bob")

    print(f"\nMagnitude of alice: {float(jnp.linalg.norm(entity_a)):.6f}")
    print(f"Magnitude of bob: {float(jnp.linalg.norm(entity_b)):.6f}")

    # Bundle them
    bundled = backend.bundle([entity_a, entity_b])

    print(f"Magnitude after bundle: {float(jnp.linalg.norm(bundled)):.6f}")

    # Check if normalization is happening
    if abs(float(jnp.linalg.norm(bundled)) - 1.0) < 0.01:
        print("INFO  Bundle is normalized (magnitude ~1.0)")
    else:
        print(f"INFO  Bundle is NOT normalized (magnitude = {float(jnp.linalg.norm(bundled)):.6f})")

    # Test: can we still recover components?
    sim_a = backend.similarity(bundled, entity_a)
    sim_b = backend.similarity(bundled, entity_b)

    print(f"\nSimilarity of bundle to alice: {float(sim_a):.6f}")
    print(f"Similarity of bundle to bob: {float(sim_b):.6f}")

    if sim_a > 0.5 and sim_b > 0.5:
        print("PASS Both components recoverable from bundle")
    else:
        print("FAIL Components NOT well-preserved in bundle")


def test_complex_similarity():
    """Verify complex similarity computation."""
    print("\n" + "="*80)
    print("SANITY CHECK 3: Complex Similarity Computation")
    print("="*80)

    backend = FHRRBackend(dim=8192, seed=42)
    registry = SymbolRegistry(dim=8192, seed=42)

    entity_a = registry.register(SymbolSpace.ENTITIES, "alice")

    # Self-similarity should be 1.0
    self_sim = backend.similarity(entity_a, entity_a)
    print(f"Self-similarity: {float(self_sim):.6f}")

    if abs(float(self_sim) - 1.0) < 0.01:
        print("PASS PASS: Self-similarity is 1.0")
    else:
        print("FAIL FAIL: Self-similarity is not 1.0!")
        print("   This indicates similarity computation issue")

    # Different entities should have low similarity
    entity_b = registry.register(SymbolSpace.ENTITIES, "bob")
    cross_sim = backend.similarity(entity_a, entity_b)
    print(f"Cross-similarity (alice vs bob): {float(cross_sim):.6f}")

    if abs(float(cross_sim)) < 0.2:
        print("PASS PASS: Random entities have low similarity")
    else:
        print("WARNING  WARNING: Random entities have unexpectedly high similarity")


def test_role_filler_cross_talk():
    """Measure actual cross-talk in role-filler binding."""
    print("\n" + "="*80)
    print("DIAGNOSTIC: Role-Filler Cross-Talk Measurement")
    print("="*80)

    backend = FHRRBackend(dim=8192, seed=42)
    registry = SymbolRegistry(dim=8192, seed=42)

    # Create a binary bundle: ARG1*alice + ARG2*bob
    entity_a = registry.register(SymbolSpace.ENTITIES, "alice")
    entity_b = registry.register(SymbolSpace.ENTITIES, "bob")
    role_1 = registry.register(SymbolSpace.ARG_ROLES, "ARG1")
    role_2 = registry.register(SymbolSpace.ARG_ROLES, "ARG2")

    bound_1 = backend.bind(role_1, entity_a)
    bound_2 = backend.bind(role_2, entity_b)

    bundle = backend.bundle([bound_1, bound_2])

    # Decode position 2 (should get bob)
    decoded_2 = backend.unbind(bundle, role_2)

    # Similarity to target (bob)
    sim_signal = backend.similarity(decoded_2, entity_b)

    # Similarity to noise source (alice)
    sim_noise = backend.similarity(decoded_2, entity_a)

    print(f"Unbind(ARG1*alice + ARG2*bob, ARG2):")
    print(f"  Similarity to bob (signal): {float(sim_signal):.6f}")
    print(f"  Similarity to alice (noise): {float(sim_noise):.6f}")
    print(f"  Signal-to-noise ratio: {float(sim_signal/max(abs(sim_noise), 0.001)):.2f}")

    if sim_signal > 0.5:
        print("PASS Signal similarity is acceptable")
    else:
        print("FAIL Signal similarity is too low!")

    return float(sim_signal)


def test_shift_vs_role():
    """Compare shift-based vs role-based for same atoms."""
    print("\n" + "="*80)
    print("COMPARISON: Shift vs Role-Filler for Binary Predicate")
    print("="*80)

    backend = FHRRBackend(dim=8192, seed=42)
    registry = SymbolRegistry(dim=8192, seed=42)

    entity_a = registry.register(SymbolSpace.ENTITIES, "alice")
    entity_b = registry.register(SymbolSpace.ENTITIES, "bob")

    # Shift-based: shift(alice,1) + shift(bob,2)
    shifted_a = backend.permute(entity_a, 1)
    shifted_b = backend.permute(entity_b, 2)
    shift_bundle = backend.bundle([shifted_a, shifted_b])

    # Decode position 2
    decoded_shift = backend.permute(shift_bundle, -2)
    sim_shift = backend.similarity(decoded_shift, entity_b)

    print(f"Shift-based encoding:")
    print(f"  Decode position 2 -> similarity to bob: {float(sim_shift):.6f}")

    # Role-based: ARG1*alice + ARG2*bob
    role_1 = registry.register(SymbolSpace.ARG_ROLES, "ARG1")
    role_2 = registry.register(SymbolSpace.ARG_ROLES, "ARG2")

    bound_1 = backend.bind(role_1, entity_a)
    bound_2 = backend.bind(role_2, entity_b)
    role_bundle = backend.bundle([bound_1, bound_2])

    # Decode position 2
    decoded_role = backend.unbind(role_bundle, role_2)
    sim_role = backend.similarity(decoded_role, entity_b)

    print(f"\nRole-filler encoding:")
    print(f"  Decode position 2 -> similarity to bob: {float(sim_role):.6f}")

    print(f"\nDifference: {float(sim_shift - sim_role):.6f}")
    print(f"Shift is {float(sim_shift/sim_role):.2f}x better")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VSA ENCODING SANITY CHECKS")
    print("Expert-recommended diagnostics")
    print("="*80)

    # Run all sanity checks
    bind_unbind_score = test_bind_unbind_identity()
    test_bundling_normalization()
    test_complex_similarity()
    role_filler_score = test_role_filler_cross_talk()
    test_shift_vs_role()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Bind/unbind identity: {bind_unbind_score:.6f} (should be >0.95)")
    print(f"Role-filler signal recovery: {role_filler_score:.6f} (you're getting ~0.31)")

    if bind_unbind_score < 0.95:
        print("\nWARNING  CRITICAL: Fix bind/unbind implementation first!")
    elif role_filler_score < 0.5:
        print("\nWARNING  Role-filler has fundamental limitations for your use case")
        print("   -> Recommend: Predicate-bound + shift encoding")
