"""Test vsax 1.3.0 unbind operator to see if it works correctly."""

import jax
import jax.numpy as jnp
from vsar.kernel.vsa_backend import FHRRBackend, MAPBackend


def test_fhrr_unbind():
    """Test FHRR backend unbind operation."""
    print("\n=== Testing FHRR Unbind ===")

    backend = FHRRBackend(dim=512, seed=42)
    key = jax.random.PRNGKey(42)

    # Generate random vectors
    key1, key2 = jax.random.split(key)
    a = backend.generate_random(key1, (512,))
    b = backend.generate_random(key2, (512,))

    # Bind: c = a * b
    c = backend.bind(a, b)

    # Unbind: b_recovered = c / a
    b_recovered = backend.unbind(c, a)

    # Check similarity
    similarity = backend.similarity(b, b_recovered)

    print(f"Original b vs Recovered b similarity: {similarity:.4f}")
    print(f"Expected: > 0.7 for working unbind")
    print(f"Result: {'PASS' if similarity > 0.7 else 'FAIL'}")

    return similarity


def test_map_unbind():
    """Test MAP backend unbind operation."""
    print("\n=== Testing MAP Unbind ===")

    backend = MAPBackend(dim=512, seed=42)
    key = jax.random.PRNGKey(42)

    # Generate random vectors
    key1, key2 = jax.random.split(key)
    a = backend.generate_random(key1, (512,))
    b = backend.generate_random(key2, (512,))

    # Bind: c = a * b
    c = backend.bind(a, b)

    # Unbind: b_recovered = c / a
    b_recovered = backend.unbind(c, a)

    # Check similarity
    similarity = backend.similarity(b, b_recovered)

    print(f"Original b vs Recovered b similarity: {similarity:.4f}")
    print(f"Expected: > 0.7 for working unbind")
    print(f"Result: {'PASS' if similarity > 0.7 else 'FAIL'}")

    return similarity


def test_multiple_seeds():
    """Test unbind with multiple random seeds for consistency."""
    print("\n=== Testing Multiple Seeds (FHRR) ===")

    seeds = [42, 123, 456, 789, 1000]
    similarities = []

    for seed in seeds:
        backend = FHRRBackend(dim=512, seed=seed)
        key = jax.random.PRNGKey(seed)

        key1, key2 = jax.random.split(key)
        a = backend.generate_random(key1, (512,))
        b = backend.generate_random(key2, (512,))

        c = backend.bind(a, b)
        b_recovered = backend.unbind(c, a)

        similarity = backend.similarity(b, b_recovered)
        similarities.append(similarity)
        print(f"Seed {seed}: similarity = {similarity:.4f}")

    avg_similarity = sum(similarities) / len(similarities)
    print(f"\nAverage similarity: {avg_similarity:.4f}")
    print(f"All > 0.7: {'PASS' if all(s > 0.7 for s in similarities) else 'FAIL'}")

    return avg_similarity


if __name__ == "__main__":
    print("Testing vsax 1.3.0 unbind operator")
    print("=" * 50)

    fhrr_sim = test_fhrr_unbind()
    map_sim = test_map_unbind()
    avg_sim = test_multiple_seeds()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"FHRR unbind similarity: {fhrr_sim:.4f}")
    print(f"MAP unbind similarity:  {map_sim:.4f}")
    print(f"Multi-seed average:     {avg_sim:.4f}")
    print()

    if fhrr_sim > 0.7 and map_sim > 0.7 and avg_sim > 0.7:
        print("SUCCESS: Unbind operator works correctly!")
        print("  We can consider reverting to bind/unbind architecture")
    else:
        print("FAILURE: Unbind operator still broken")
        print("  Keep current shift-based approach")
