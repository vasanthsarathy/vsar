"""Unit tests for knowledge base persistence."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from vsar.kb.persistence import load_kb, save_kb
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend


class TestKBPersistence:
    """Test cases for KB save/load functionality."""

    @pytest.fixture
    def backend(self) -> FHRRBackend:
        """Create test backend."""
        return FHRRBackend(dim=128, seed=42)

    @pytest.fixture
    def kb_with_facts(self, backend: FHRRBackend) -> KnowledgeBase:
        """Create KB with sample facts."""
        kb = KnowledgeBase(backend)

        # Add some facts
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec3 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        kb.insert("parent", vec2, ("bob", "carol"))
        kb.insert("sibling", vec3, ("bob", "carol"))

        return kb

    @pytest.fixture
    def temp_file(self, tmp_path: Path) -> Path:
        """Create temporary file path."""
        return tmp_path / "test_kb.h5"

    def test_save_creates_file(self, kb_with_facts: KnowledgeBase, temp_file: Path) -> None:
        """Test that save creates the file."""
        assert not temp_file.exists()
        save_kb(kb_with_facts, temp_file)
        assert temp_file.exists()

    def test_save_creates_parent_directories(
        self, kb_with_facts: KnowledgeBase, tmp_path: Path
    ) -> None:
        """Test that save creates parent directories if needed."""
        nested_path = tmp_path / "nested" / "dirs" / "kb.h5"
        assert not nested_path.parent.exists()

        save_kb(kb_with_facts, nested_path)
        assert nested_path.exists()

    def test_save_and_load_roundtrip(
        self, kb_with_facts: KnowledgeBase, backend: FHRRBackend, temp_file: Path
    ) -> None:
        """Test that save/load preserves all data."""
        # Save
        save_kb(kb_with_facts, temp_file)

        # Load
        loaded_kb = load_kb(backend, temp_file)

        # Verify counts
        assert loaded_kb.count() == kb_with_facts.count()
        assert loaded_kb.count("parent") == kb_with_facts.count("parent")
        assert loaded_kb.count("sibling") == kb_with_facts.count("sibling")

        # Verify predicates
        assert set(loaded_kb.predicates()) == set(kb_with_facts.predicates())

    def test_save_and_load_preserves_facts(
        self, kb_with_facts: KnowledgeBase, backend: FHRRBackend, temp_file: Path
    ) -> None:
        """Test that facts are preserved correctly."""
        save_kb(kb_with_facts, temp_file)
        loaded_kb = load_kb(backend, temp_file)

        # Verify facts
        original_parent_facts = kb_with_facts.get_facts("parent")
        loaded_parent_facts = loaded_kb.get_facts("parent")
        assert loaded_parent_facts == original_parent_facts

        original_sibling_facts = kb_with_facts.get_facts("sibling")
        loaded_sibling_facts = loaded_kb.get_facts("sibling")
        assert loaded_sibling_facts == original_sibling_facts

    def test_save_and_load_preserves_bundles(
        self, kb_with_facts: KnowledgeBase, backend: FHRRBackend, temp_file: Path
    ) -> None:
        """Test that bundles are preserved correctly."""
        save_kb(kb_with_facts, temp_file)
        loaded_kb = load_kb(backend, temp_file)

        # Verify vectors
        original_parent_vectors = kb_with_facts.get_vectors("parent")
        loaded_parent_vectors = loaded_kb.get_vectors("parent")
        assert len(original_parent_vectors) > 0
        assert len(loaded_parent_vectors) > 0
        assert len(original_parent_vectors) == len(loaded_parent_vectors)
        for orig, loaded in zip(original_parent_vectors, loaded_parent_vectors, strict=False):
            assert jnp.allclose(loaded, orig)

        original_sibling_vectors = kb_with_facts.get_vectors("sibling")
        loaded_sibling_vectors = loaded_kb.get_vectors("sibling")
        assert len(original_sibling_vectors) > 0
        assert len(loaded_sibling_vectors) > 0
        assert len(original_sibling_vectors) == len(loaded_sibling_vectors)
        for orig, loaded in zip(original_sibling_vectors, loaded_sibling_vectors, strict=False):
            assert jnp.allclose(loaded, orig)

    def test_save_empty_kb(self, backend: FHRRBackend, temp_file: Path) -> None:
        """Test saving empty KB."""
        empty_kb = KnowledgeBase(backend)
        save_kb(empty_kb, temp_file)

        loaded_kb = load_kb(backend, temp_file)
        assert loaded_kb.count() == 0
        assert loaded_kb.predicates() == []

    def test_load_nonexistent_file_raises_error(self, backend: FHRRBackend, tmp_path: Path) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_kb(backend, tmp_path / "nonexistent.h5")

    def test_save_overwrites_existing_file(
        self, kb_with_facts: KnowledgeBase, backend: FHRRBackend, temp_file: Path
    ) -> None:
        """Test that saving overwrites existing file."""
        # Save first version
        save_kb(kb_with_facts, temp_file)

        # Create new KB with different facts
        new_kb = KnowledgeBase(backend)
        vec = backend.generate_random(jax.random.PRNGKey(0), (backend.dimensionension,))
        new_kb.insert("human", vec, ("alice",))

        # Overwrite
        save_kb(new_kb, temp_file)

        # Load and verify new content
        loaded_kb = load_kb(backend, temp_file)
        assert loaded_kb.count() == 1
        assert loaded_kb.has_predicate("human")
        assert not loaded_kb.has_predicate("parent")

    def test_load_different_backend_same_dimension(
        self, kb_with_facts: KnowledgeBase, temp_file: Path
    ) -> None:
        """Test loading KB with different backend instance."""
        save_kb(kb_with_facts, temp_file)

        # Create new backend with same dimension
        new_backend = FHRRBackend(dim=128, seed=99)
        loaded_kb = load_kb(new_backend, temp_file)

        # Should load successfully
        assert loaded_kb.count() == kb_with_facts.count()
        assert loaded_kb.backend == new_backend

    def test_save_and_load_preserves_fact_order(
        self, backend: FHRRBackend, temp_file: Path
    ) -> None:
        """Test that fact insertion order is preserved."""
        kb = KnowledgeBase(backend)

        # Insert facts in specific order
        for i in range(5):
            vec = backend.generate_random(jax.random.PRNGKey(0), (backend.dimensionension,))
            kb.insert("test", vec, (f"entity_{i}",))

        save_kb(kb, temp_file)
        loaded_kb = load_kb(backend, temp_file)

        # Verify order
        original_facts = kb.get_facts("test")
        loaded_facts = loaded_kb.get_facts("test")
        assert loaded_facts == original_facts

    def test_save_multiple_predicates(self, backend: FHRRBackend, temp_file: Path) -> None:
        """Test saving KB with many predicates."""
        kb = KnowledgeBase(backend)

        # Add facts for multiple predicates
        for i in range(10):
            vec = backend.generate_random(jax.random.PRNGKey(0), (backend.dimensionension,))
            kb.insert(f"predicate_{i}", vec, (f"arg_{i}",))

        save_kb(kb, temp_file)
        loaded_kb = load_kb(backend, temp_file)

        assert loaded_kb.count() == 10
        assert len(loaded_kb.predicates()) == 10
        for i in range(10):
            assert loaded_kb.has_predicate(f"predicate_{i}")

    def test_save_with_string_path(
        self, kb_with_facts: KnowledgeBase, backend: FHRRBackend, tmp_path: Path
    ) -> None:
        """Test that save/load work with string paths."""
        temp_file = str(tmp_path / "test_kb.h5")

        save_kb(kb_with_facts, temp_file)
        loaded_kb = load_kb(backend, temp_file)

        assert loaded_kb.count() == kb_with_facts.count()

    def test_save_with_pathlib_path(
        self, kb_with_facts: KnowledgeBase, backend: FHRRBackend, temp_file: Path
    ) -> None:
        """Test that save/load work with Path objects."""
        save_kb(kb_with_facts, temp_file)
        loaded_kb = load_kb(backend, temp_file)

        assert loaded_kb.count() == kb_with_facts.count()
