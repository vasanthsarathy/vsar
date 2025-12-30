"""Unit tests for role vector management."""

import jax.numpy as jnp
import pytest

from vsar.encoding.roles import RoleVectorManager
from vsar.kernel.vsa_backend import FHRRBackend


class TestRoleVectorManager:
    """Test cases for RoleVectorManager."""

    @pytest.fixture
    def backend(self) -> FHRRBackend:
        """Create test backend."""
        return FHRRBackend(dim=128, seed=42)

    @pytest.fixture
    def manager(self, backend: FHRRBackend) -> RoleVectorManager:
        """Create test role manager."""
        return RoleVectorManager(backend, seed=42)

    def test_initialization(self, backend: FHRRBackend) -> None:
        """Test role manager initialization."""
        manager = RoleVectorManager(backend, seed=42)
        assert manager.backend == backend
        assert manager.seed == 42
        assert len(manager._roles) == 0

    def test_get_role_creates_vector(self, manager: RoleVectorManager) -> None:
        """Test that getting a role creates a hypervector."""
        role1 = manager.get_role(1)
        assert role1 is not None
        assert role1.shape == (128,)

    def test_get_role_is_deterministic(self, manager: RoleVectorManager) -> None:
        """Test that getting same role twice returns same vector."""
        role1_a = manager.get_role(1)
        role1_b = manager.get_role(1)

        assert jnp.allclose(role1_a, role1_b)

    def test_get_role_is_normalized(self, manager: RoleVectorManager) -> None:
        """Test that role vectors have unit norm."""
        role1 = manager.get_role(1)
        norm = jnp.linalg.norm(role1)
        assert jnp.abs(norm - 1.0) < 1e-5

    def test_different_roles_are_different(self, manager: RoleVectorManager) -> None:
        """Test that different roles get different vectors."""
        role1 = manager.get_role(1)
        role2 = manager.get_role(2)

        # Should be different
        similarity = manager.backend.similarity(role1, role2)
        assert similarity < 0.9

    def test_get_role_position_validation(self, manager: RoleVectorManager) -> None:
        """Test that position must be >= 1."""
        with pytest.raises(ValueError, match="Position must be >= 1"):
            manager.get_role(0)

        with pytest.raises(ValueError, match="Position must be >= 1"):
            manager.get_role(-1)

    def test_get_roles_returns_list(self, manager: RoleVectorManager) -> None:
        """Test getting multiple roles at once."""
        roles = manager.get_roles(3)
        assert len(roles) == 3
        assert all(role.shape == (128,) for role in roles)

    def test_get_roles_are_consistent(self, manager: RoleVectorManager) -> None:
        """Test that get_roles returns same vectors as get_role."""
        roles = manager.get_roles(3)

        assert jnp.allclose(roles[0], manager.get_role(1))
        assert jnp.allclose(roles[1], manager.get_role(2))
        assert jnp.allclose(roles[2], manager.get_role(3))

    def test_clear(self, manager: RoleVectorManager) -> None:
        """Test clearing cached role vectors."""
        manager.get_role(1)
        manager.get_role(2)
        assert len(manager._roles) == 2

        manager.clear()
        assert len(manager._roles) == 0

    def test_clear_does_not_affect_determinism(self, manager: RoleVectorManager) -> None:
        """Test that clearing cache doesn't change vectors."""
        role1_before = manager.get_role(1)
        manager.clear()
        role1_after = manager.get_role(1)

        assert jnp.allclose(role1_before, role1_after)

    def test_similarity_matrix(self, manager: RoleVectorManager) -> None:
        """Test similarity matrix computation."""
        sim_matrix = manager.similarity_matrix(3)

        # Check shape
        assert sim_matrix.shape == (3, 3)

        # Diagonal should be ~1.0 (self-similarity)
        assert sim_matrix[0, 0] > 0.99
        assert sim_matrix[1, 1] > 0.99
        assert sim_matrix[2, 2] > 0.99

    def test_similarity_matrix_off_diagonal(self, manager: RoleVectorManager) -> None:
        """Test that role vectors are dissimilar."""
        sim_matrix = manager.similarity_matrix(3)

        # Off-diagonal should be low (dissimilar)
        # Using 0.6 threshold to account for approximate VSA properties
        assert abs(sim_matrix[0, 1]) < 0.6
        assert abs(sim_matrix[0, 2]) < 0.6
        assert abs(sim_matrix[1, 2]) < 0.6

    def test_different_seeds_produce_different_roles(self, backend: FHRRBackend) -> None:
        """Test that different seeds produce different role vectors."""
        manager1 = RoleVectorManager(backend, seed=42)
        manager2 = RoleVectorManager(backend, seed=123)

        role1_a = manager1.get_role(1)
        role1_b = manager2.get_role(1)

        # Should be different
        assert not jnp.allclose(role1_a, role1_b)

    def test_same_seed_produces_same_roles(self, backend: FHRRBackend) -> None:
        """Test that same seed produces identical role vectors."""
        manager1 = RoleVectorManager(backend, seed=42)
        manager2 = RoleVectorManager(backend, seed=42)

        role1_a = manager1.get_role(1)
        role1_b = manager2.get_role(1)

        # Should be identical
        assert jnp.allclose(role1_a, role1_b)

    def test_large_position_numbers(self, manager: RoleVectorManager) -> None:
        """Test that large position numbers work correctly."""
        role100 = manager.get_role(100)
        role1000 = manager.get_role(1000)

        assert role100.shape == (128,)
        assert role1000.shape == (128,)

        # Should be different
        similarity = manager.backend.similarity(role100, role1000)
        assert similarity < 0.9
