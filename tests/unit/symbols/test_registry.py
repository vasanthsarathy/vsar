"""Unit tests for symbol registry."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class TestSymbolRegistry:
    """Test cases for SymbolRegistry."""

    @pytest.fixture
    def backend(self) -> FHRRBackend:
        """Create test backend."""
        return FHRRBackend(dim=128, seed=42)

    @pytest.fixture
    def registry(self, backend: FHRRBackend) -> SymbolRegistry:
        """Create test registry."""
        return SymbolRegistry(backend, seed=42)

    def test_initialization(self, backend: FHRRBackend) -> None:
        """Test registry initialization."""
        registry = SymbolRegistry(backend, seed=42)
        assert registry.backend == backend
        assert registry.seed == 42
        assert registry.count() == 0

    def test_register_creates_vector(self, registry: SymbolRegistry) -> None:
        """Test that registering a symbol creates a hypervector."""
        vec = registry.register(SymbolSpace.ENTITIES, "alice")
        assert vec is not None
        assert vec.shape == (128,)

    def test_register_is_deterministic(self, registry: SymbolRegistry) -> None:
        """Test that registering same symbol twice returns same vector."""
        vec1 = registry.register(SymbolSpace.ENTITIES, "alice")
        vec2 = registry.register(SymbolSpace.ENTITIES, "alice")

        assert jnp.allclose(vec1, vec2)

    def test_register_different_symbols(self, registry: SymbolRegistry) -> None:
        """Test that different symbols get different vectors."""
        alice = registry.register(SymbolSpace.ENTITIES, "alice")
        bob = registry.register(SymbolSpace.ENTITIES, "bob")

        # Should be different
        similarity = registry.backend.similarity(alice, bob)
        assert similarity < 0.9

    def test_register_same_name_different_spaces(
        self, registry: SymbolRegistry
    ) -> None:
        """Test that same name in different spaces gets different vectors."""
        entity_alice = registry.register(SymbolSpace.ENTITIES, "alice")
        relation_alice = registry.register(SymbolSpace.RELATIONS, "alice")

        similarity = registry.backend.similarity(entity_alice, relation_alice)
        assert similarity < 0.9

    def test_get_existing_symbol(self, registry: SymbolRegistry) -> None:
        """Test getting an existing symbol."""
        original = registry.register(SymbolSpace.ENTITIES, "alice")
        retrieved = registry.get(SymbolSpace.ENTITIES, "alice")

        assert retrieved is not None
        assert jnp.allclose(retrieved, original)

    def test_get_nonexistent_symbol(self, registry: SymbolRegistry) -> None:
        """Test getting a nonexistent symbol returns None."""
        result = registry.get(SymbolSpace.ENTITIES, "nonexistent")
        assert result is None

    def test_cleanup_finds_exact_match(self, registry: SymbolRegistry) -> None:
        """Test that cleanup finds exact symbol match."""
        alice = registry.register(SymbolSpace.ENTITIES, "alice")
        bob = registry.register(SymbolSpace.ENTITIES, "bob")

        # Cleanup with alice's exact vector should return alice as top match
        results = registry.cleanup(SymbolSpace.ENTITIES, alice, k=2)

        assert len(results) > 0
        best_match, score = results[0]
        assert best_match == "alice"
        assert score > 0.99  # Should be near perfect match

    def test_cleanup_ranks_by_similarity(self, registry: SymbolRegistry) -> None:
        """Test that cleanup ranks results by similarity."""
        alice = registry.register(SymbolSpace.ENTITIES, "alice")
        registry.register(SymbolSpace.ENTITIES, "bob")
        registry.register(SymbolSpace.ENTITIES, "carol")

        # Slightly perturbed alice vector
        noisy_alice = alice + jnp.ones_like(alice) * 0.01
        noisy_alice = registry.backend.normalize(noisy_alice)

        results = registry.cleanup(SymbolSpace.ENTITIES, noisy_alice, k=3)

        # Alice should still be top match
        assert results[0][0] == "alice"
        # Scores should be descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_cleanup_respects_space_boundaries(
        self, registry: SymbolRegistry
    ) -> None:
        """Test that cleanup only searches within specified space."""
        entity_alice = registry.register(SymbolSpace.ENTITIES, "alice")
        registry.register(SymbolSpace.RELATIONS, "parent")

        # Cleanup in ENTITIES space should only find entities
        results = registry.cleanup(SymbolSpace.ENTITIES, entity_alice, k=10)

        assert all(name != "parent" for name, _ in results)

    def test_cleanup_empty_space(self, registry: SymbolRegistry) -> None:
        """Test cleanup on empty space returns empty list."""
        vec = jnp.ones(128)
        results = registry.cleanup(SymbolSpace.ENTITIES, vec, k=10)

        assert results == []

    def test_cleanup_respects_k_parameter(self, registry: SymbolRegistry) -> None:
        """Test that cleanup returns at most k results."""
        for i in range(10):
            registry.register(SymbolSpace.ENTITIES, f"entity_{i}")

        vec = registry.get(SymbolSpace.ENTITIES, "entity_0")
        assert vec is not None

        results = registry.cleanup(SymbolSpace.ENTITIES, vec, k=5)
        assert len(results) <= 5

    def test_symbols_returns_all_names(self, registry: SymbolRegistry) -> None:
        """Test that symbols() returns all registered names."""
        registry.register(SymbolSpace.ENTITIES, "alice")
        registry.register(SymbolSpace.ENTITIES, "bob")
        registry.register(SymbolSpace.RELATIONS, "parent")

        all_symbols = registry.symbols()
        assert len(all_symbols) == 3
        assert "alice" in all_symbols
        assert "bob" in all_symbols
        assert "parent" in all_symbols

    def test_symbols_filtered_by_space(self, registry: SymbolRegistry) -> None:
        """Test that symbols() can be filtered by space."""
        registry.register(SymbolSpace.ENTITIES, "alice")
        registry.register(SymbolSpace.ENTITIES, "bob")
        registry.register(SymbolSpace.RELATIONS, "parent")

        entities = registry.symbols(SymbolSpace.ENTITIES)
        assert len(entities) == 2
        assert "alice" in entities
        assert "bob" in entities
        assert "parent" not in entities

    def test_count_all_symbols(self, registry: SymbolRegistry) -> None:
        """Test counting all symbols."""
        registry.register(SymbolSpace.ENTITIES, "alice")
        registry.register(SymbolSpace.ENTITIES, "bob")
        registry.register(SymbolSpace.RELATIONS, "parent")

        assert registry.count() == 3

    def test_count_by_space(self, registry: SymbolRegistry) -> None:
        """Test counting symbols by space."""
        registry.register(SymbolSpace.ENTITIES, "alice")
        registry.register(SymbolSpace.ENTITIES, "bob")
        registry.register(SymbolSpace.RELATIONS, "parent")

        assert registry.count(SymbolSpace.ENTITIES) == 2
        assert registry.count(SymbolSpace.RELATIONS) == 1
        assert registry.count(SymbolSpace.ATTRIBUTES) == 0

    def test_save_and_load(
        self, registry: SymbolRegistry, tmp_path: Path
    ) -> None:
        """Test save/load roundtrip."""
        # Register some symbols
        registry.register(SymbolSpace.ENTITIES, "alice")
        registry.register(SymbolSpace.ENTITIES, "bob")
        registry.register(SymbolSpace.RELATIONS, "parent")

        # Save
        save_path = tmp_path / "registry.h5"
        registry.save(save_path)

        # Create new registry and load
        new_backend = FHRRBackend(dim=128, seed=42)
        new_registry = SymbolRegistry(new_backend, seed=42)
        new_registry.load(save_path)

        # Verify symbols are preserved
        assert new_registry.count() == 3
        assert "alice" in new_registry.symbols(SymbolSpace.ENTITIES)
        assert "bob" in new_registry.symbols(SymbolSpace.ENTITIES)
        assert "parent" in new_registry.symbols(SymbolSpace.RELATIONS)

    def test_clear(self, registry: SymbolRegistry) -> None:
        """Test clearing the registry."""
        registry.register(SymbolSpace.ENTITIES, "alice")
        registry.register(SymbolSpace.ENTITIES, "bob")

        assert registry.count() == 2

        registry.clear()

        assert registry.count() == 0
        assert registry.symbols() == []

    def test_different_seeds_produce_different_registries(
        self, backend: FHRRBackend
    ) -> None:
        """Test that different seeds produce different vectors."""
        registry1 = SymbolRegistry(backend, seed=42)
        registry2 = SymbolRegistry(backend, seed=123)

        vec1 = registry1.register(SymbolSpace.ENTITIES, "alice")
        vec2 = registry2.register(SymbolSpace.ENTITIES, "alice")

        # Should be different
        assert not jnp.allclose(vec1, vec2)

    def test_same_seed_produces_same_vectors(self, backend: FHRRBackend) -> None:
        """Test that same seed produces identical vectors across registries."""
        registry1 = SymbolRegistry(backend, seed=42)
        registry2 = SymbolRegistry(backend, seed=42)

        vec1 = registry1.register(SymbolSpace.ENTITIES, "alice")
        vec2 = registry2.register(SymbolSpace.ENTITIES, "alice")

        # Should be identical
        assert jnp.allclose(vec1, vec2)
