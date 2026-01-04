"""Tests for VSAX primitive operations (Phase 0).

This test suite validates that VSAX bind/unbind operations work correctly
and meet the success criteria for Phase 0:
- Bind/unbind reconstruction error < 0.01
- Cleanup success rate > 99% for SNR > 0.5
- All primitives pass deterministic tests with fixed seeds
"""

import jax
import jax.numpy as jnp
import pytest
from vsar.kernel.vsa_backend import FHRRBackend


class TestBindUnbindIdentity:
    """Test that (a ⊗ b) ⊘ b ≈ a."""

    def test_bind_unbind_identity_basic(self):
        """Basic test: bind then unbind should recover original vector."""
        backend = FHRRBackend(dim=512, seed=42)
        key = jax.random.PRNGKey(42)

        # Generate two random vectors
        key, subkey1, subkey2 = jax.random.split(key, 3)
        a = backend.generate_random(subkey1, (512,))
        b = backend.generate_random(subkey2, (512,))

        # Bind: c = a ⊗ b
        c = backend.bind(a, b)

        # Unbind: a_recovered = c ⊘ b
        a_recovered = backend.unbind(c, b)

        # Check similarity (correct metric for VSA)
        similarity = backend.similarity(a, a_recovered)

        assert similarity > 0.99, f"Similarity {similarity:.4f} below threshold 0.99"

    def test_bind_unbind_identity_multiple_seeds(self):
        """Test bind/unbind with different random seeds."""
        seeds = [0, 1, 42, 123, 999]

        for seed in seeds:
            backend = FHRRBackend(dim=512, seed=seed)
            key = jax.random.PRNGKey(seed)

            key, subkey1, subkey2 = jax.random.split(key, 3)
            a = backend.generate_random(subkey1, (512,))
            b = backend.generate_random(subkey2, (512,))

            c = backend.bind(a, b)
            a_recovered = backend.unbind(c, b)

            similarity = backend.similarity(a, a_recovered)

            assert similarity > 0.99, f"Seed {seed}: similarity {similarity:.4f} below threshold 0.99"

    def test_bind_unbind_identity_high_dimensions(self):
        """Test bind/unbind with higher dimensions."""
        dims = [512, 1024, 2048]

        for dim in dims:
            backend = FHRRBackend(dim=dim, seed=42)
            key = jax.random.PRNGKey(42)

            key, subkey1, subkey2 = jax.random.split(key, 3)
            a = backend.generate_random(subkey1, (dim,))
            b = backend.generate_random(subkey2, (dim,))

            c = backend.bind(a, b)
            a_recovered = backend.unbind(c, b)

            similarity = backend.similarity(a, a_recovered)

            assert similarity > 0.99, f"Dim {dim}: similarity {similarity:.4f} below threshold 0.99"


class TestBindCommutativity:
    """Test that a ⊗ b == b ⊗ a."""

    def test_bind_is_commutative(self):
        """Verify bind operation is commutative."""
        backend = FHRRBackend(dim=512, seed=42)
        key = jax.random.PRNGKey(42)

        key, subkey1, subkey2 = jax.random.split(key, 3)
        a = backend.generate_random(subkey1, (512,))
        b = backend.generate_random(subkey2, (512,))

        # Bind in both orders
        ab = backend.bind(a, b)
        ba = backend.bind(b, a)

        # Check they are equal (within floating point precision)
        diff = jnp.linalg.norm(ab - ba)

        assert diff < 1e-4, f"Bind not commutative: ||a⊗b - b⊗a|| = {diff:.8f}"


class TestUnbindInverse:
    """Test that a ⊘ b == a ⊗ conj(b)."""

    def test_unbind_equals_bind_conjugate(self):
        """Verify unbind is equivalent to binding with inverse."""
        backend = FHRRBackend(dim=512, seed=42)
        key = jax.random.PRNGKey(42)

        key, subkey1, subkey2 = jax.random.split(key, 3)
        a = backend.generate_random(subkey1, (512,))
        b = backend.generate_random(subkey2, (512,))

        # Method 1: Use unbind
        result_unbind = backend.unbind(a, b)

        # Method 2: Use bind with inverse
        b_inv = backend._model.opset.inverse(b)
        result_bind_inv = backend.bind(a, b_inv)
        result_bind_inv = backend.normalize(result_bind_inv)

        # Check equivalence
        diff = jnp.linalg.norm(result_unbind - result_bind_inv)

        assert diff < 1e-5, f"Unbind not equivalent to bind(a, inv(b)): diff = {diff:.8f}"


class TestBundleSuperposition:
    """Test that (a ⊕ b ⊕ c) contains signals for a, b, c."""

    def test_bundle_contains_components(self):
        """Verify bundled vector has high similarity to components."""
        backend = FHRRBackend(dim=512, seed=42)
        key = jax.random.PRNGKey(42)

        # Generate 3 random vectors
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        a = backend.generate_random(subkey1, (512,))
        b = backend.generate_random(subkey2, (512,))
        c = backend.generate_random(subkey3, (512,))

        # Bundle them
        bundled = backend.bundle([a, b, c])

        # Check similarity to each component
        sim_a = backend.similarity(bundled, a)
        sim_b = backend.similarity(bundled, b)
        sim_c = backend.similarity(bundled, c)

        # Each component should be detectable in the bundle
        # With 3 components, similarity typically around 0.5-0.7
        assert sim_a > 0.3, f"Similarity to a ({sim_a:.3f}) too low"
        assert sim_b > 0.3, f"Similarity to b ({sim_b:.3f}) too low"
        assert sim_c > 0.3, f"Similarity to c ({sim_c:.3f}) too low"

    def test_bundle_many_components(self):
        """Test bundling with many components (capacity test)."""
        backend = FHRRBackend(dim=1024, seed=42)
        key = jax.random.PRNGKey(42)

        # Generate 10 random vectors
        vectors = []
        for i in range(10):
            key, subkey = jax.random.split(key)
            vec = backend.generate_random(subkey, (1024,))
            vectors.append(vec)

        # Bundle them
        bundled = backend.bundle(vectors)

        # Check similarity to each component
        similarities = [backend.similarity(bundled, vec) for vec in vectors]
        avg_sim = sum(similarities) / len(similarities)

        # With 10 components, average similarity should still be reasonable
        assert avg_sim > 0.2, f"Average similarity ({avg_sim:.3f}) too low for bundle capacity"


class TestCleanupUnderNoise:
    """Test that cleanup recovers correct symbol from noisy vector."""

    def test_cleanup_perfect_match(self):
        """Test cleanup with exact match."""
        backend = FHRRBackend(dim=512, seed=42)
        key = jax.random.PRNGKey(42)

        # Create a "codebook" of 10 symbols
        codebook = {}
        for i in range(10):
            key, subkey = jax.random.split(key)
            symbol_name = f"symbol_{i}"
            symbol_vec = backend.generate_random(subkey, (512,))
            codebook[symbol_name] = symbol_vec

        # Query with exact symbol
        query = codebook["symbol_5"]

        # Find nearest neighbor
        best_match = None
        best_sim = -1
        for name, vec in codebook.items():
            sim = backend.similarity(query, vec)
            if sim > best_sim:
                best_sim = sim
                best_match = name

        assert best_match == "symbol_5", f"Cleanup failed: matched {best_match} instead of symbol_5"
        assert best_sim > 0.99, f"Perfect match similarity ({best_sim:.4f}) not close to 1.0"

    def test_cleanup_with_noise(self):
        """Test cleanup with controlled noise (SNR > 0.5)."""
        backend = FHRRBackend(dim=1024, seed=42)
        key = jax.random.PRNGKey(42)

        # Create a codebook of 20 symbols
        codebook = {}
        for i in range(20):
            key, subkey = jax.random.split(key)
            symbol_name = f"symbol_{i}"
            symbol_vec = backend.generate_random(subkey, (1024,))
            codebook[symbol_name] = symbol_vec

        # Test cleanup for each symbol with noise
        target_symbol = "symbol_7"
        clean_vec = codebook[target_symbol]

        # Add noise to achieve SNR > 0.5
        key, noise_key = jax.random.split(key)
        noise = backend.generate_random(noise_key, (1024,))

        # Create noisy vector: 70% signal, 30% noise (SNR ≈ 0.7)
        noisy_vec = 0.7 * clean_vec + 0.3 * noise
        noisy_vec = backend.normalize(noisy_vec)

        # Cleanup: find nearest neighbor
        best_match = None
        best_sim = -1
        for name, vec in codebook.items():
            sim = backend.similarity(noisy_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_match = name

        assert best_match == target_symbol, f"Cleanup with noise failed: matched {best_match} instead of {target_symbol}"

    def test_cleanup_success_rate(self):
        """Test that cleanup succeeds > 99% of time with SNR > 0.5."""
        backend = FHRRBackend(dim=1024, seed=42)
        key = jax.random.PRNGKey(42)

        # Create a codebook
        codebook = {}
        for i in range(50):
            key, subkey = jax.random.split(key)
            symbol_name = f"symbol_{i}"
            symbol_vec = backend.generate_random(subkey, (1024,))
            codebook[symbol_name] = symbol_vec

        # Run 100 trials
        num_trials = 100
        num_successes = 0

        for trial in range(num_trials):
            # Pick a random symbol
            target_idx = trial % 50
            target_symbol = f"symbol_{target_idx}"
            clean_vec = codebook[target_symbol]

            # Add noise (SNR ≈ 0.6)
            key, noise_key = jax.random.split(key)
            noise = backend.generate_random(noise_key, (1024,))
            noisy_vec = 0.6 * clean_vec + 0.4 * noise
            noisy_vec = backend.normalize(noisy_vec)

            # Cleanup
            best_match = None
            best_sim = -1
            for name, vec in codebook.items():
                sim = backend.similarity(noisy_vec, vec)
                if sim > best_sim:
                    best_sim = sim
                    best_match = name

            if best_match == target_symbol:
                num_successes += 1

        success_rate = num_successes / num_trials

        assert success_rate > 0.99, f"Cleanup success rate ({success_rate:.2%}) below 99% threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
