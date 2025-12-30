"""Unit tests for unbinding operations."""

import jax
import jax.numpy as jnp
import pytest

from vsar.encoding.roles import RoleVectorManager
from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.retrieval.unbind import (
    extract_variable_binding,
    unbind_query_from_bundle,
    unbind_role,
)
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class TestUnbindOperations:
    """Test cases for unbinding operations."""

    @pytest.fixture
    def backend(self) -> FHRRBackend:
        """Create test backend."""
        return FHRRBackend(dim=128, seed=42)

    @pytest.fixture
    def registry(self, backend: FHRRBackend) -> SymbolRegistry:
        """Create test registry."""
        return SymbolRegistry(backend, seed=42)

    @pytest.fixture
    def encoder(self, backend: FHRRBackend, registry: SymbolRegistry) -> VSAEncoder:
        """Create test encoder."""
        return VSAEncoder(backend, registry, seed=42)

    @pytest.fixture
    def role_manager(self, backend: FHRRBackend) -> RoleVectorManager:
        """Create test role manager."""
        return RoleVectorManager(backend, seed=42)

    def test_unbind_query_from_bundle(self, backend: FHRRBackend, encoder: VSAEncoder) -> None:
        """Test unbinding query from bundle."""
        # Encode a fact
        atom_vec = encoder.encode_atom("parent", ["alice", "bob"])

        # Encode query
        query_vec = encoder.encode_query("parent", ["alice", None])

        # Unbind query from bundle (which is just the atom)
        result = unbind_query_from_bundle(query_vec, atom_vec, backend)

        assert result is not None
        assert result.shape == (128,)
        assert jnp.abs(jnp.linalg.norm(result) - 1.0) < 1e-5  # Normalized

    def test_unbind_role(self, backend: FHRRBackend, role_manager: RoleVectorManager) -> None:
        """Test unbinding role vector."""
        # Create role-filler binding
        role1 = role_manager.get_role(1)
        entity_vec = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        entity_vec = backend.normalize(entity_vec)

        # Bind role to entity
        role_filler = backend.bind(role1, entity_vec)

        # Unbind role
        recovered = unbind_role(role_filler, role1, backend)

        assert recovered is not None
        assert recovered.shape == (128,)

        # Should be similar to original entity (approximate)
        similarity = backend.similarity(recovered, entity_vec)
        assert similarity > 0.4  # Approximate recovery

    def test_extract_variable_binding(
        self,
        backend: FHRRBackend,
        encoder: VSAEncoder,
        role_manager: RoleVectorManager,
    ) -> None:
        """Test extracting variable binding from bundle."""
        # Encode a fact: parent(alice, bob)
        atom_vec = encoder.encode_atom("parent", ["alice", "bob"])

        # Encode query: parent(alice, X)
        query_vec = encoder.encode_query("parent", ["alice", None])

        # Get role for position 2 (the variable)
        role2 = role_manager.get_role(2)

        # Extract variable binding
        entity_vec = extract_variable_binding(atom_vec, query_vec, role2, backend)

        assert entity_vec is not None
        assert entity_vec.shape == (128,)

        # Should be similar to bob's encoding (approximate)
        bob_vec = encoder.registry.get(SymbolSpace.ENTITIES, "bob")
        if bob_vec is not None:
            similarity = backend.similarity(entity_vec, bob_vec)
            # Might be low due to approximate nature of VSA
            assert similarity > 0.2

    def test_unbind_preserves_normalization(self, backend: FHRRBackend) -> None:
        """Test that unbind operations preserve normalization."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(1), (backend.dimension,))

        vec1 = backend.normalize(vec1)
        vec2 = backend.normalize(vec2)

        # Unbind
        result = backend.unbind(vec1, vec2)

        # Check normalization (backend.unbind should normalize)
        # Actually, our unbind functions normalize, so test those
        result_norm = jnp.linalg.norm(result)
        # Allow some tolerance
        assert jnp.abs(result_norm - 1.0) < 0.1
