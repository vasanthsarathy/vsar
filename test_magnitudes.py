"""Debug magnitude issues in interference cancellation."""

import jax.numpy as jnp
from vsar.encoding.role_filler_encoder import RoleFillerEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


def test_magnitude_tracking():
    """Track magnitudes through encoding/decoding pipeline."""
    print("\n" + "="*80)
    print("MAGNITUDE TRACKING TEST")
    print("="*80)

    backend = FHRRBackend(dim=8192, seed=42)
    registry = SymbolRegistry(dim=8192, seed=42)
    encoder = RoleFillerEncoder(backend, registry, seed=42)

    # Get entity vectors
    alice = registry.register(SymbolSpace.ENTITIES, "alice")
    bob = registry.register(SymbolSpace.ENTITIES, "bob")
    pred = registry.register(SymbolSpace.PREDICATES, "parent")

    print(f"\n1. Entity vectors (fresh from registry):")
    print(f"   |alice| = {float(jnp.linalg.norm(alice)):.6f}")
    print(f"   |bob|   = {float(jnp.linalg.norm(bob)):.6f}")

    # Shift them
    alice_shifted = backend.permute(alice, 1)
    bob_shifted = backend.permute(bob, 2)

    print(f"\n2. Shifted entity vectors:")
    print(f"   |shift(alice,1)| = {float(jnp.linalg.norm(alice_shifted)):.6f}")
    print(f"   |shift(bob,2)|   = {float(jnp.linalg.norm(bob_shifted)):.6f}")

    # Bundle them (now WITHOUT normalization)
    args_bundle = backend.bundle([alice_shifted, bob_shifted])

    print(f"\n3. Args bundle (sum of shifted vectors, no normalization):")
    print(f"   |args_bundle| = {float(jnp.linalg.norm(args_bundle)):.6f}")
    print(f"   Expected: √2 ≈ 1.414 (sum of two unit vectors)")

    # Bind with predicate
    atom = backend.bind(pred, args_bundle)

    print(f"\n4. After binding with predicate:")
    print(f"   |atom (before final norm)| = {float(jnp.linalg.norm(atom)):.6f}")

    # Normalize (as encoder does)
    atom_normalized = backend.normalize(atom)

    print(f"\n5. After final normalization (what gets stored):")
    print(f"   |atom (stored)| = {float(jnp.linalg.norm(atom_normalized)):.6f}")

    # Unbind predicate
    args_bundle_recovered = backend.unbind(atom_normalized, pred)

    print(f"\n6. After unbinding predicate from stored atom:")
    print(f"   |args_bundle_recovered| = {float(jnp.linalg.norm(args_bundle_recovered)):.6f}")
    print(f"   Original |args_bundle| = {float(jnp.linalg.norm(args_bundle)):.6f}")

    magnitude_ratio = float(jnp.linalg.norm(args_bundle_recovered)) / float(jnp.linalg.norm(args_bundle))
    print(f"   Magnitude ratio: {magnitude_ratio:.6f}")

    # Try cancellation with recovered bundle
    print(f"\n7. Interference cancellation attempt:")
    print(f"   Subtracting fresh shift(alice,1) from recovered bundle...")

    # Fresh shifted alice
    alice_shifted_fresh = backend.permute(alice, 1)
    print(f"   |shift(alice,1) fresh| = {float(jnp.linalg.norm(alice_shifted_fresh)):.6f}")

    # Subtract
    cleaned_bundle = args_bundle_recovered - alice_shifted_fresh
    print(f"   |cleaned_bundle| = {float(jnp.linalg.norm(cleaned_bundle)):.6f}")

    # Decode position 2
    decoded = backend.permute(cleaned_bundle, -2)
    similarity = backend.similarity(decoded, bob)

    print(f"\n8. Final result:")
    print(f"   Similarity to bob: {similarity:.6f}")
    print(f"   Expected: ~0.95-1.0 if cancellation perfect")

    # What if we scale the fresh contribution to match?
    print(f"\n9. Try scaling the contribution to match magnitudes:")
    alice_shifted_scaled = alice_shifted_fresh * magnitude_ratio
    cleaned_bundle_scaled = args_bundle_recovered - alice_shifted_scaled
    decoded_scaled = backend.permute(cleaned_bundle_scaled, -2)
    similarity_scaled = backend.similarity(decoded_scaled, bob)

    print(f"   Scaled contribution by {magnitude_ratio:.6f}")
    print(f"   Similarity to bob: {similarity_scaled:.6f}")


if __name__ == "__main__":
    test_magnitude_tracking()
