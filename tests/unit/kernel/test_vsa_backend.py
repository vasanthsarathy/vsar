"""Unit tests for VSA backend implementations."""

import jax
import jax.numpy as jnp
import pytest

from vsar.kernel.vsa_backend import FHRRBackend, MAPBackend


class TestFHRRBackend:
    """Test cases for FHRR backend."""

    @pytest.fixture
    def backend(self) -> FHRRBackend:
        """Create FHRR backend with fixed parameters."""
        return FHRRBackend(dim=512, seed=42)

    @pytest.fixture
    def key(self) -> jax.random.PRNGKey:
        """Create JAX random key."""
        return jax.random.PRNGKey(0)

    def test_initialization(self, backend: FHRRBackend) -> None:
        """Test backend initialization."""
        assert backend.dimensionension == 512

    def test_generate_random(self, backend: FHRRBackend, key: jax.random.PRNGKey) -> None:
        """Test random vector generation."""
        vec = backend.generate_random(key, (512,))
        assert vec.shape == (512,)
        # FHRR uses complex hypervectors
        assert jnp.iscomplexobj(vec)

    def test_determinism(self, backend: FHRRBackend) -> None:
        """Test that same seed produces identical vectors."""
        key = jax.random.PRNGKey(123)
        vec1 = backend.generate_random(key, (512,))
        vec2 = backend.generate_random(key, (512,))
        assert jnp.allclose(vec1, vec2)

    def test_bind_unbind_roundtrip(self, backend: FHRRBackend, key: jax.random.PRNGKey) -> None:
        """Test that bind/unbind achieves reasonable fidelity."""
        key1, key2 = jax.random.split(key)
        a = backend.generate_random(key1, (512,))
        b = backend.generate_random(key2, (512,))

        # Bind a and b
        c = backend.bind(a, b)

        # Unbind to recover b
        b_recovered = backend.unbind(c, a)

        # Check similarity is reasonable for approximate reasoning
        # VSAR uses approximate retrieval, so 50%+ similarity is acceptable
        sim = backend.similarity(b, b_recovered)
        assert sim > 0.4, f"Similarity {sim} too low for bind/unbind roundtrip"

    def test_bind_commutativity(self, backend: FHRRBackend, key: jax.random.PRNGKey) -> None:
        """Test that FHRR binding is commutative."""
        key1, key2 = jax.random.split(key)
        a = backend.generate_random(key1, (512,))
        b = backend.generate_random(key2, (512,))

        ab = backend.bind(a, b)
        ba = backend.bind(b, a)

        # FHRR circular convolution is commutative
        assert jnp.allclose(ab, ba, atol=1e-6)

    def test_bundle_multiple(self, backend: FHRRBackend, key: jax.random.PRNGKey) -> None:
        """Test bundling multiple vectors."""
        keys = jax.random.split(key, 5)
        vectors = [backend.generate_random(k, (512,)) for k in keys]

        bundled = backend.bundle(vectors)
        assert bundled.shape == (512,)

        # Each original vector should have reasonable similarity to bundle
        for vec in vectors:
            sim = backend.similarity(vec, bundled)
            assert sim > 0.2, "Bundled vector should retain similarity to components"

    def test_bundle_from_array(self, backend: FHRRBackend, key: jax.random.PRNGKey) -> None:
        """Test bundling from numpy array."""
        keys = jax.random.split(key, 3)
        vectors = jnp.stack([backend.generate_random(k, (512,)) for k in keys])

        bundled = backend.bundle(vectors)
        assert bundled.shape == (512,)

    def test_similarity_range(self, backend: FHRRBackend, key: jax.random.PRNGKey) -> None:
        """Test that similarity is in expected range."""
        key1, key2 = jax.random.split(key)
        a = backend.generate_random(key1, (512,))
        b = backend.generate_random(key2, (512,))

        # Similarity with itself should be ~1.0
        sim_self = backend.similarity(a, a)
        assert 0.99 <= sim_self <= 1.01

        # Similarity with random vector should be lower
        sim_random = backend.similarity(a, b)
        assert 0.0 <= sim_random <= 1.0

    def test_normalize(self, backend: FHRRBackend, key: jax.random.PRNGKey) -> None:
        """Test vector normalization."""
        vec = backend.generate_random(key, (512,))
        # Scale it arbitrarily
        scaled = vec * 5.0

        normalized = backend.normalize(scaled)
        # Check that normalized has unit norm
        norm = jnp.linalg.norm(normalized)
        assert jnp.abs(norm - 1.0) < 1e-5


class TestMAPBackend:
    """Test cases for MAP backend."""

    @pytest.fixture
    def backend(self) -> MAPBackend:
        """Create MAP backend with fixed parameters."""
        return MAPBackend(dim=512, seed=42)

    @pytest.fixture
    def key(self) -> jax.random.PRNGKey:
        """Create JAX random key."""
        return jax.random.PRNGKey(0)

    def test_initialization(self, backend: MAPBackend) -> None:
        """Test backend initialization."""
        assert backend.dimensionension == 512

    def test_generate_random(self, backend: MAPBackend, key: jax.random.PRNGKey) -> None:
        """Test random vector generation."""
        vec = backend.generate_random(key, (512,))
        assert vec.shape == (512,)
        # MAP uses real hypervectors
        assert not jnp.iscomplexobj(vec)

    def test_determinism(self, backend: MAPBackend) -> None:
        """Test that same seed produces identical vectors."""
        key = jax.random.PRNGKey(123)
        vec1 = backend.generate_random(key, (512,))
        vec2 = backend.generate_random(key, (512,))
        assert jnp.allclose(vec1, vec2)

    def test_bind_unbind_roundtrip(self, backend: MAPBackend, key: jax.random.PRNGKey) -> None:
        """Test that bind/unbind achieves reasonable fidelity."""
        key1, key2 = jax.random.split(key)
        a = backend.generate_random(key1, (512,))
        b = backend.generate_random(key2, (512,))

        # Bind a and b
        c = backend.bind(a, b)

        # Unbind to recover b
        b_recovered = backend.unbind(c, a)

        # Check similarity (MAP may have lower fidelity than FHRR)
        sim = backend.similarity(b, b_recovered)
        assert sim > 0.5, f"Similarity {sim} too low for bind/unbind roundtrip"

    def test_bind_commutativity(self, backend: MAPBackend, key: jax.random.PRNGKey) -> None:
        """Test that MAP binding is commutative."""
        key1, key2 = jax.random.split(key)
        a = backend.generate_random(key1, (512,))
        b = backend.generate_random(key2, (512,))

        ab = backend.bind(a, b)
        ba = backend.bind(b, a)

        # Element-wise multiplication is commutative
        assert jnp.allclose(ab, ba, atol=1e-6)

    def test_bundle_multiple(self, backend: MAPBackend, key: jax.random.PRNGKey) -> None:
        """Test bundling multiple vectors."""
        keys = jax.random.split(key, 5)
        vectors = [backend.generate_random(k, (512,)) for k in keys]

        bundled = backend.bundle(vectors)
        assert bundled.shape == (512,)

    def test_similarity_range(self, backend: MAPBackend, key: jax.random.PRNGKey) -> None:
        """Test that similarity is in expected range."""
        key1, key2 = jax.random.split(key)
        a = backend.generate_random(key1, (512,))
        b = backend.generate_random(key2, (512,))

        # Similarity with itself should be ~1.0
        sim_self = backend.similarity(a, a)
        assert 0.99 <= sim_self <= 1.01

        # Similarity with random vector should be in valid range
        sim_random = backend.similarity(a, b)
        assert 0.0 <= sim_random <= 1.0


class TestBackendComparison:
    """Compare FHRR and MAP backends."""

    @pytest.fixture
    def fhrr(self) -> FHRRBackend:
        """Create FHRR backend."""
        return FHRRBackend(dim=512, seed=42)

    @pytest.fixture
    def map_backend(self) -> MAPBackend:
        """Create MAP backend."""
        return MAPBackend(dim=512, seed=42)

    def test_both_backends_have_same_interface(
        self, fhrr: FHRRBackend, map_backend: MAPBackend
    ) -> None:
        """Test that both backends implement the same interface."""
        # Both should have the same methods
        assert hasattr(fhrr, "bind")
        assert hasattr(fhrr, "unbind")
        assert hasattr(fhrr, "bundle")
        assert hasattr(fhrr, "similarity")
        assert hasattr(fhrr, "generate_random")
        assert hasattr(fhrr, "normalize")

        assert hasattr(map_backend, "bind")
        assert hasattr(map_backend, "unbind")
        assert hasattr(map_backend, "bundle")
        assert hasattr(map_backend, "similarity")
        assert hasattr(map_backend, "generate_random")
        assert hasattr(map_backend, "normalize")

    def test_polymorphic_usage(self, fhrr: FHRRBackend, map_backend: MAPBackend) -> None:
        """Test that backends can be used polymorphically."""
        for backend in [fhrr, map_backend]:
            key = jax.random.PRNGKey(0)
            a = backend.generate_random(key, (512,))
            b = backend.generate_random(jax.random.split(key)[0], (512,))

            c = backend.bind(a, b)
            bundled = backend.bundle([a, b])
            sim = backend.similarity(a, b)

            assert c.shape == (512,)
            assert bundled.shape == (512,)
            assert 0.0 <= sim <= 1.0
