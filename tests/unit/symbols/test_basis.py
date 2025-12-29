"""Unit tests for basis vector generation and persistence."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.basis import generate_basis, load_basis, save_basis
from vsar.symbols.spaces import SymbolSpace


class TestGenerateBasis:
    """Test cases for basis vector generation."""

    @pytest.fixture
    def backend(self) -> FHRRBackend:
        """Create test backend."""
        return FHRRBackend(dim=128, seed=42)

    def test_deterministic_generation(self, backend: FHRRBackend) -> None:
        """Test that same inputs produce identical vectors."""
        vec1 = generate_basis(SymbolSpace.ENTITIES, "alice", backend, seed=42)
        vec2 = generate_basis(SymbolSpace.ENTITIES, "alice", backend, seed=42)

        assert jnp.allclose(vec1, vec2)

    def test_different_names_produce_different_vectors(
        self, backend: FHRRBackend
    ) -> None:
        """Test that different names produce different vectors."""
        alice = generate_basis(SymbolSpace.ENTITIES, "alice", backend, seed=42)
        bob = generate_basis(SymbolSpace.ENTITIES, "bob", backend, seed=42)

        # Vectors should be different
        similarity = backend.similarity(alice, bob)
        assert similarity < 0.9  # Not too similar

    def test_different_spaces_produce_different_vectors(
        self, backend: FHRRBackend
    ) -> None:
        """Test that same name in different spaces produces different vectors."""
        entity_alice = generate_basis(SymbolSpace.ENTITIES, "alice", backend, seed=42)
        relation_alice = generate_basis(SymbolSpace.RELATIONS, "alice", backend, seed=42)

        # Should be different despite same name
        similarity = backend.similarity(entity_alice, relation_alice)
        assert similarity < 0.9

    def test_different_seeds_produce_different_vectors(
        self, backend: FHRRBackend
    ) -> None:
        """Test that different seeds produce different vectors."""
        vec1 = generate_basis(SymbolSpace.ENTITIES, "alice", backend, seed=42)
        vec2 = generate_basis(SymbolSpace.ENTITIES, "alice", backend, seed=123)

        # Should be different with different seeds
        assert not jnp.allclose(vec1, vec2)

    def test_generated_vectors_are_normalized(self, backend: FHRRBackend) -> None:
        """Test that generated vectors have unit norm."""
        vec = generate_basis(SymbolSpace.ENTITIES, "alice", backend, seed=42)

        norm = jnp.linalg.norm(vec)
        assert jnp.abs(norm - 1.0) < 1e-5

    def test_vector_has_correct_dimension(self, backend: FHRRBackend) -> None:
        """Test that generated vector has correct dimension."""
        vec = generate_basis(SymbolSpace.ENTITIES, "alice", backend, seed=42)

        assert vec.shape == (128,)


class TestBasisPersistence:
    """Test cases for basis save/load functionality."""

    @pytest.fixture
    def backend(self) -> FHRRBackend:
        """Create test backend."""
        return FHRRBackend(dim=128, seed=42)

    @pytest.fixture
    def sample_basis(self, backend: FHRRBackend) -> dict:
        """Create sample basis for testing."""
        return {
            (SymbolSpace.ENTITIES, "alice"): generate_basis(
                SymbolSpace.ENTITIES, "alice", backend, 42
            ),
            (SymbolSpace.ENTITIES, "bob"): generate_basis(
                SymbolSpace.ENTITIES, "bob", backend, 42
            ),
            (SymbolSpace.RELATIONS, "parent"): generate_basis(
                SymbolSpace.RELATIONS, "parent", backend, 42
            ),
        }

    @pytest.fixture
    def temp_file(self, tmp_path: Path) -> Path:
        """Create temporary file path."""
        return tmp_path / "test_basis.h5"

    def test_save_and_load_roundtrip(
        self, sample_basis: dict, temp_file: Path
    ) -> None:
        """Test that save/load preserves all data."""
        save_basis(temp_file, sample_basis)
        loaded_basis = load_basis(temp_file)

        # Check all keys are present
        assert set(loaded_basis.keys()) == set(sample_basis.keys())

        # Check all vectors are identical
        for key in sample_basis:
            assert jnp.allclose(loaded_basis[key], sample_basis[key])

    def test_save_creates_file(self, sample_basis: dict, temp_file: Path) -> None:
        """Test that save creates the file."""
        assert not temp_file.exists()
        save_basis(temp_file, sample_basis)
        assert temp_file.exists()

    def test_save_creates_parent_directories(
        self, sample_basis: dict, tmp_path: Path
    ) -> None:
        """Test that save creates parent directories if needed."""
        nested_path = tmp_path / "nested" / "dirs" / "basis.h5"
        assert not nested_path.parent.exists()

        save_basis(nested_path, sample_basis)
        assert nested_path.exists()

    def test_load_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_basis(tmp_path / "nonexistent.h5")

    def test_save_empty_basis(self, temp_file: Path) -> None:
        """Test saving empty basis."""
        empty_basis: dict = {}
        save_basis(temp_file, empty_basis)
        loaded = load_basis(temp_file)

        assert len(loaded) == 0

    def test_loaded_basis_preserves_spaces(
        self, sample_basis: dict, temp_file: Path
    ) -> None:
        """Test that symbol spaces are preserved correctly."""
        save_basis(temp_file, sample_basis)
        loaded_basis = load_basis(temp_file)

        for (space, name) in loaded_basis.keys():
            assert isinstance(space, SymbolSpace)
            assert isinstance(name, str)

    def test_multiple_symbols_same_space(
        self, backend: FHRRBackend, temp_file: Path
    ) -> None:
        """Test saving/loading multiple symbols in same space."""
        basis = {
            (SymbolSpace.ENTITIES, "alice"): generate_basis(
                SymbolSpace.ENTITIES, "alice", backend, 42
            ),
            (SymbolSpace.ENTITIES, "bob"): generate_basis(
                SymbolSpace.ENTITIES, "bob", backend, 42
            ),
            (SymbolSpace.ENTITIES, "carol"): generate_basis(
                SymbolSpace.ENTITIES, "carol", backend, 42
            ),
        }

        save_basis(temp_file, basis)
        loaded = load_basis(temp_file)

        assert len(loaded) == 3
        for key in basis:
            assert key in loaded
