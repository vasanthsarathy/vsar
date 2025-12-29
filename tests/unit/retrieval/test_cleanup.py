"""Unit tests for cleanup operations."""

import jax
import jax.numpy as jnp
import pytest

from vsar.kernel.vsa_backend import FHRRBackend
from vsar.retrieval.cleanup import batch_cleanup, cleanup, get_top_symbol
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class TestCleanup:
    """Test cases for cleanup operations."""

    @pytest.fixture
    def backend(self) -> FHRRBackend:
        """Create test backend."""
        return FHRRBackend(dim=128, seed=42)

    @pytest.fixture
    def registry(self, backend: FHRRBackend) -> SymbolRegistry:
        """Create test registry with sample entities."""
        registry = SymbolRegistry(backend, seed=42)

        # Register some entities
        registry.register(SymbolSpace.ENTITIES, "alice")
        registry.register(SymbolSpace.ENTITIES, "bob")
        registry.register(SymbolSpace.ENTITIES, "carol")
        registry.register(SymbolSpace.ENTITIES, "dave")

        return registry

    def test_cleanup_exact_match(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test cleanup with exact vector match."""
        # Get exact vector for alice
        alice_vec = registry.get(SymbolSpace.ENTITIES, "alice")
        assert alice_vec is not None

        # Cleanup should return alice as top match
        results = cleanup(SymbolSpace.ENTITIES, alice_vec, registry, backend, k=5)

        assert len(results) > 0
        assert results[0][0] == "alice"
        assert results[0][1] > 0.99  # Near perfect similarity

    def test_cleanup_noisy_vector(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test cleanup with noisy vector."""
        # Get alice's vector and add noise
        alice_vec = registry.get(SymbolSpace.ENTITIES, "alice")
        assert alice_vec is not None

        noise = backend.generate_random(
            jax.random.PRNGKey(999), (backend.dimension,)
        )
        noise = backend.normalize(noise) * 0.1  # Small noise

        noisy_vec = alice_vec + noise
        noisy_vec = backend.normalize(noisy_vec)

        # Cleanup should still return alice as top match
        results = cleanup(SymbolSpace.ENTITIES, noisy_vec, registry, backend, k=5)

        assert len(results) > 0
        assert results[0][0] == "alice"

    def test_cleanup_returns_top_k(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test that cleanup returns at most k results."""
        alice_vec = registry.get(SymbolSpace.ENTITIES, "alice")
        assert alice_vec is not None

        results = cleanup(SymbolSpace.ENTITIES, alice_vec, registry, backend, k=2)

        assert len(results) <= 2

    def test_cleanup_sorted_by_similarity(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test that cleanup results are sorted by similarity."""
        alice_vec = registry.get(SymbolSpace.ENTITIES, "alice")
        assert alice_vec is not None

        results = cleanup(SymbolSpace.ENTITIES, alice_vec, registry, backend, k=4)

        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_cleanup_empty_space(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test cleanup on empty symbol space."""
        vec = backend.generate_random(
            jax.random.PRNGKey(0), (backend.dimension,)
        )
        vec = backend.normalize(vec)

        # ATTRIBUTES space is empty
        results = cleanup(SymbolSpace.ATTRIBUTES, vec, registry, backend, k=10)

        assert results == []

    def test_cleanup_respects_space_boundaries(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test that cleanup only searches within specified space."""
        # Register a relation
        registry.register(SymbolSpace.RELATIONS, "parent")

        # Get entity vector
        alice_vec = registry.get(SymbolSpace.ENTITIES, "alice")
        assert alice_vec is not None

        # Cleanup in ENTITIES should not return "parent"
        results = cleanup(SymbolSpace.ENTITIES, alice_vec, registry, backend, k=10)

        entity_names = [name for name, _ in results]
        assert "parent" not in entity_names

    def test_batch_cleanup(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test batch cleanup on multiple vectors."""
        alice_vec = registry.get(SymbolSpace.ENTITIES, "alice")
        bob_vec = registry.get(SymbolSpace.ENTITIES, "bob")
        assert alice_vec is not None
        assert bob_vec is not None

        vectors = [alice_vec, bob_vec]
        results = batch_cleanup(
            SymbolSpace.ENTITIES, vectors, registry, backend, k=3
        )

        assert len(results) == 2
        assert results[0][0][0] == "alice"  # First result, top match
        assert results[1][0][0] == "bob"  # Second result, top match

    def test_batch_cleanup_empty_list(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test batch cleanup with empty list."""
        results = batch_cleanup(
            SymbolSpace.ENTITIES, [], registry, backend, k=5
        )

        assert results == []

    def test_get_top_symbol_returns_best_match(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test get_top_symbol returns single best match."""
        alice_vec = registry.get(SymbolSpace.ENTITIES, "alice")
        assert alice_vec is not None

        result = get_top_symbol(
            SymbolSpace.ENTITIES, alice_vec, registry, backend
        )

        assert result is not None
        assert result[0] == "alice"
        assert result[1] > 0.99

    def test_get_top_symbol_empty_space(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test get_top_symbol on empty space returns None."""
        vec = backend.generate_random(
            jax.random.PRNGKey(0), (backend.dimension,)
        )
        vec = backend.normalize(vec)

        result = get_top_symbol(
            SymbolSpace.ATTRIBUTES, vec, registry, backend
        )

        assert result is None

    def test_cleanup_with_similar_vectors(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test cleanup can distinguish between similar vectors."""
        alice_vec = registry.get(SymbolSpace.ENTITIES, "alice")
        bob_vec = registry.get(SymbolSpace.ENTITIES, "bob")
        assert alice_vec is not None
        assert bob_vec is not None

        # Use alice's vector
        results = cleanup(SymbolSpace.ENTITIES, alice_vec, registry, backend, k=4)

        # Alice should be top, not bob
        assert results[0][0] == "alice"
        assert results[0][1] > results[1][1]  # Alice has higher score
