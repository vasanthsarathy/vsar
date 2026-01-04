"""Tests for symbol spaces, typed codebooks, and symbol registry (Phase 1.1).

This test suite validates:
- Symbol space enum
- TypedCodebook for managing symbols within a space
- SymbolRegistry for managing all codebooks
- Typed cleanup operations
"""

import jax
import jax.numpy as jnp
import pytest

from vsar.symbols.spaces import SymbolSpace
from vsar.symbols.codebook import TypedCodebook
from vsar.symbols.registry import SymbolRegistry


class TestSymbolSpaces:
    """Test the SymbolSpace enum."""

    def test_all_spaces_defined(self):
        """Verify all 11 symbol spaces are defined."""
        expected_spaces = {
            "ENTITIES", "CONCEPTS", "ROLES", "FUNCTIONS", "PREDICATES",
            "ARG_ROLES", "STRUCT_ROLES", "TAGS", "OPS", "EPI_OPS", "GRAPH_OPS"
        }
        actual_spaces = {space.name for space in SymbolSpace}
        assert actual_spaces == expected_spaces

    def test_space_values(self):
        """Verify symbol space abbreviations."""
        assert SymbolSpace.ENTITIES.value == "E"
        assert SymbolSpace.CONCEPTS.value == "C"
        assert SymbolSpace.PREDICATES.value == "P"
        assert SymbolSpace.ARG_ROLES.value == "ARG"
        assert SymbolSpace.TAGS.value == "TAG"


class TestTypedCodebook:
    """Test TypedCodebook functionality."""

    def test_create_codebook(self):
        """Test creating a typed codebook."""
        codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512, seed=42)
        assert codebook.space == SymbolSpace.ENTITIES
        assert codebook.dim == 512
        assert len(codebook) == 0

    def test_register_symbol(self):
        """Test registering a symbol."""
        codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512, seed=42)

        alice = codebook.register("alice")

        assert alice.shape == (512,)
        assert jnp.iscomplexobj(alice)
        assert len(codebook) == 1
        assert "alice" in codebook

    def test_register_multiple_symbols(self):
        """Test registering multiple symbols."""
        codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512, seed=42)

        alice = codebook.register("alice")
        bob = codebook.register("bob")
        carol = codebook.register("carol")

        assert len(codebook) == 3
        assert "alice" in codebook
        assert "bob" in codebook
        assert "carol" in codebook

        # Vectors should be different (nearly orthogonal)
        sim_ab = jnp.abs(jnp.sum(alice * jnp.conj(bob))) / (jnp.linalg.norm(alice) * jnp.linalg.norm(bob))
        assert sim_ab < 0.2  # Low similarity for random vectors

    def test_register_idempotent(self):
        """Test that registering same symbol twice returns same vector."""
        codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512, seed=42)

        alice1 = codebook.register("alice")
        alice2 = codebook.register("alice")

        assert jnp.allclose(alice1, alice2)
        assert len(codebook) == 1

    def test_get_registered_symbol(self):
        """Test getting a registered symbol."""
        codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512, seed=42)

        alice_registered = codebook.register("alice")
        alice_retrieved = codebook.get("alice")

        assert alice_retrieved is not None
        assert jnp.allclose(alice_registered, alice_retrieved)

    def test_get_unregistered_symbol(self):
        """Test getting an unregistered symbol returns None."""
        codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512, seed=42)
        result = codebook.get("bob")
        assert result is None

    def test_cleanup_exact_match(self):
        """Test cleanup with exact match."""
        codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512, seed=42)

        alice = codebook.register("alice")
        bob = codebook.register("bob")

        # Cleanup with exact vector
        results = codebook.cleanup(alice, k=1)

        assert len(results) == 1
        assert results[0][0] == "alice"
        assert results[0][1] > 0.99  # Very high similarity

    def test_cleanup_top_k(self):
        """Test cleanup returning top-k results."""
        codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512, seed=42)

        for name in ["alice", "bob", "carol", "dave", "eve"]:
            codebook.register(name)

        alice = codebook.get("alice")

        # Get top 3
        results = codebook.cleanup(alice, k=3)

        assert len(results) == 3
        assert results[0][0] == "alice"  # Best match first
        assert results[0][1] > 0.99

    def test_cleanup_with_threshold(self):
        """Test cleanup with similarity threshold."""
        codebook = TypedCodebook(SymbolSpace.ENTITIES, dim=512, seed=42)

        for name in ["alice", "bob", "carol"]:
            codebook.register(name)

        alice = codebook.get("alice")

        # High threshold should only return alice
        results = codebook.cleanup(alice, k=10, threshold=0.95)

        assert len(results) >= 1
        assert results[0][0] == "alice"


class TestSymbolRegistry:
    """Test SymbolRegistry functionality."""

    def test_create_registry(self):
        """Test creating a symbol registry."""
        registry = SymbolRegistry(dim=512, seed=42)

        assert registry.dim == 512
        assert registry.symbol_count() == 0

    def test_register_in_different_spaces(self):
        """Test registering symbols in different spaces."""
        registry = SymbolRegistry(dim=512, seed=42)

        alice = registry.register(SymbolSpace.ENTITIES, "alice")
        parent = registry.register(SymbolSpace.PREDICATES, "parent")
        arg1 = registry.register(SymbolSpace.ARG_ROLES, "ARG1")

        assert registry.symbol_count() == 3
        assert registry.symbol_count(SymbolSpace.ENTITIES) == 1
        assert registry.symbol_count(SymbolSpace.PREDICATES) == 1
        assert registry.symbol_count(SymbolSpace.ARG_ROLES) == 1

    def test_same_name_different_spaces(self):
        """Test that same name in different spaces gets different vectors."""
        registry = SymbolRegistry(dim=512, seed=42)

        # Register "parent" in both PREDICATES and ENTITIES
        parent_pred = registry.register(SymbolSpace.PREDICATES, "parent")
        parent_entity = registry.register(SymbolSpace.ENTITIES, "parent")

        # Vectors should be different (from different codebooks)
        sim = jnp.abs(jnp.sum(parent_pred * jnp.conj(parent_entity))) / (
            jnp.linalg.norm(parent_pred) * jnp.linalg.norm(parent_entity)
        )
        assert sim < 0.2  # Should be nearly orthogonal

    def test_get_from_registry(self):
        """Test getting symbols from registry."""
        registry = SymbolRegistry(dim=512, seed=42)

        alice_registered = registry.register(SymbolSpace.ENTITIES, "alice")
        alice_retrieved = registry.get(SymbolSpace.ENTITIES, "alice")

        assert alice_retrieved is not None
        assert jnp.allclose(alice_registered, alice_retrieved)

    def test_contains(self):
        """Test checking if symbol is in registry."""
        registry = SymbolRegistry(dim=512, seed=42)

        registry.register(SymbolSpace.ENTITIES, "alice")

        assert registry.contains(SymbolSpace.ENTITIES, "alice")
        assert not registry.contains(SymbolSpace.ENTITIES, "bob")
        assert not registry.contains(SymbolSpace.PREDICATES, "alice")

    def test_typed_cleanup(self):
        """Test typed cleanup: search only in specific space."""
        registry = SymbolRegistry(dim=512, seed=42)

        # Register symbols in ENTITIES
        alice = registry.register(SymbolSpace.ENTITIES, "alice")
        bob = registry.register(SymbolSpace.ENTITIES, "bob")

        # Register symbols in PREDICATES
        parent = registry.register(SymbolSpace.PREDICATES, "parent")

        # Cleanup in ENTITIES space
        results = registry.cleanup(SymbolSpace.ENTITIES, alice, k=2)

        # Should only find alice and bob, not parent
        names = [name for name, _ in results]
        assert "alice" in names
        assert "parent" not in names

    def test_get_codebook(self):
        """Test getting a specific codebook."""
        registry = SymbolRegistry(dim=512, seed=42)

        entities = registry.get_codebook(SymbolSpace.ENTITIES)

        assert isinstance(entities, TypedCodebook)
        assert entities.space == SymbolSpace.ENTITIES

        # Can use codebook directly
        alice = entities.register("alice")
        assert alice.shape == (512,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
