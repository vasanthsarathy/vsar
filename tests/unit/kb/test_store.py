"""Unit tests for knowledge base storage."""

import jax
import jax.numpy as jnp
import pytest
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend


class TestKnowledgeBase:
    """Test cases for KnowledgeBase."""

    @pytest.fixture
    def backend(self) -> FHRRBackend:
        """Create test backend."""
        return FHRRBackend(dim=128, seed=42)

    @pytest.fixture
    def kb(self, backend: FHRRBackend) -> KnowledgeBase:
        """Create test knowledge base."""
        return KnowledgeBase(backend)

    def test_initialization(self, backend: FHRRBackend) -> None:
        """Test KB initialization."""
        kb = KnowledgeBase(backend)
        assert kb.backend == backend
        assert kb.count() == 0
        assert kb.predicates() == []

    def test_insert_single_fact(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test inserting a single fact."""
        vec = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        kb.insert("parent", vec, ("alice", "bob"))

        assert kb.count() == 1
        assert kb.count("parent") == 1
        assert kb.has_predicate("parent")

    def test_insert_multiple_facts_same_predicate(
        self, kb: KnowledgeBase, backend: FHRRBackend
    ) -> None:
        """Test inserting multiple facts for same predicate."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        kb.insert("parent", vec2, ("bob", "carol"))

        assert kb.count("parent") == 2
        assert kb.count() == 2

    def test_insert_multiple_predicates(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test inserting facts for different predicates."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        kb.insert("sibling", vec2, ("bob", "carol"))

        assert kb.count() == 2
        assert kb.count("parent") == 1
        assert kb.count("sibling") == 1
        assert len(kb.predicates()) == 2

    def test_get_vectors_existing_predicate(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test getting vectors for existing predicate."""
        vec = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        kb.insert("parent", vec, ("alice", "bob"))

        vectors = kb.get_vectors("parent")
        assert len(vectors) == 1
        assert vectors[0].shape == (128,)

    def test_get_vectors_nonexistent_predicate(self, kb: KnowledgeBase) -> None:
        """Test getting vectors for nonexistent predicate returns empty list."""
        vectors = kb.get_vectors("nonexistent")
        assert len(vectors) == 0

    def test_get_vectors_accumulates(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test that vectors list accumulates multiple atoms."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(1), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        vectors1 = kb.get_vectors("parent")
        assert len(vectors1) == 1

        kb.insert("parent", vec2, ("bob", "carol"))
        vectors2 = kb.get_vectors("parent")

        # List should have grown
        assert len(vectors2) == 2
        assert jnp.allclose(vectors2[0], vec1)
        assert jnp.allclose(vectors2[1], vec2)

    def test_get_facts_existing_predicate(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test getting facts for existing predicate."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        kb.insert("parent", vec2, ("bob", "carol"))

        facts = kb.get_facts("parent")
        assert len(facts) == 2
        assert ("alice", "bob") in facts
        assert ("bob", "carol") in facts

    def test_get_facts_nonexistent_predicate(self, kb: KnowledgeBase) -> None:
        """Test getting facts for nonexistent predicate returns empty list."""
        facts = kb.get_facts("nonexistent")
        assert facts == []

    def test_predicates_list(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test getting list of all predicates."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec3 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        kb.insert("sibling", vec2, ("bob", "carol"))
        kb.insert("human", vec3, ("alice",))

        predicates = kb.predicates()
        assert len(predicates) == 3
        assert "parent" in predicates
        assert "sibling" in predicates
        assert "human" in predicates

    def test_count_all_facts(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test counting all facts across predicates."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec3 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        kb.insert("parent", vec2, ("bob", "carol"))
        kb.insert("sibling", vec3, ("bob", "carol"))

        assert kb.count() == 3
        assert kb.count("parent") == 2
        assert kb.count("sibling") == 1

    def test_count_empty_kb(self, kb: KnowledgeBase) -> None:
        """Test counting facts in empty KB."""
        assert kb.count() == 0
        assert kb.count("nonexistent") == 0

    def test_has_predicate_true(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test has_predicate returns True for existing predicate."""
        vec = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        kb.insert("parent", vec, ("alice", "bob"))

        assert kb.has_predicate("parent")

    def test_has_predicate_false(self, kb: KnowledgeBase) -> None:
        """Test has_predicate returns False for nonexistent predicate."""
        assert not kb.has_predicate("nonexistent")

    def test_clear(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test clearing all facts from KB."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        kb.insert("sibling", vec2, ("bob", "carol"))
        assert kb.count() == 2

        kb.clear()

        assert kb.count() == 0
        assert kb.predicates() == []

    def test_clear_predicate(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test clearing facts for specific predicate."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        kb.insert("sibling", vec2, ("bob", "carol"))
        assert kb.count() == 2

        kb.clear_predicate("parent")

        assert kb.count() == 1
        assert not kb.has_predicate("parent")
        assert kb.has_predicate("sibling")

    def test_clear_predicate_nonexistent(self, kb: KnowledgeBase) -> None:
        """Test clearing nonexistent predicate doesn't raise error."""
        kb.clear_predicate("nonexistent")  # Should not raise

    def test_facts_preserve_order(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test that facts are stored in insertion order."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec3 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        kb.insert("parent", vec2, ("bob", "carol"))
        kb.insert("parent", vec3, ("carol", "dave"))

        facts = kb.get_facts("parent")
        assert facts[0] == ("alice", "bob")
        assert facts[1] == ("bob", "carol")
        assert facts[2] == ("carol", "dave")

    def test_different_arity_facts(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test storing facts with different arities."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec3 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        kb.insert("human", vec1, ("alice",))
        kb.insert("parent", vec2, ("alice", "bob"))
        kb.insert("gave", vec3, ("alice", "bob", "book"))

        assert kb.get_facts("human") == [("alice",)]
        assert kb.get_facts("parent") == [("alice", "bob")]
        assert kb.get_facts("gave") == [("alice", "bob", "book")]

    def test_contains_similar_identical_fact(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test contains_similar returns True for identical fact."""
        # Generate a vector
        vec = backend.generate_random(jax.random.PRNGKey(42), (backend.dimension,))

        # Insert fact
        kb.insert("parent", vec, ("alice", "bob"))

        # Check if same vector is considered similar
        assert kb.contains_similar("parent", vec, threshold=0.95)

    def test_contains_similar_very_similar_fact(
        self, kb: KnowledgeBase, backend: FHRRBackend
    ) -> None:
        """Test contains_similar returns True for very similar fact."""
        # Generate two vectors from same seed (will be identical)
        vec1 = backend.generate_random(jax.random.PRNGKey(42), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(42), (backend.dimension,))

        # Insert first fact
        kb.insert("parent", vec1, ("alice", "bob"))

        # Check if similar vector is detected
        assert kb.contains_similar("parent", vec2, threshold=0.95)

    def test_contains_similar_novel_fact(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test contains_similar returns False for novel fact."""
        # Generate two different vectors
        vec1 = backend.generate_random(jax.random.PRNGKey(1), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(2), (backend.dimension,))

        # Insert first fact
        kb.insert("parent", vec1, ("alice", "bob"))

        # Check if different vector is considered novel
        # With random vectors, similarity should be low
        is_similar = kb.contains_similar("parent", vec2, threshold=0.95)

        # Random vectors typically have very low similarity
        # This assertion may occasionally fail due to randomness, but very unlikely
        assert not is_similar

    def test_contains_similar_nonexistent_predicate(
        self, kb: KnowledgeBase, backend: FHRRBackend
    ) -> None:
        """Test contains_similar returns False for nonexistent predicate."""
        vec = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))

        # Check nonexistent predicate
        assert not kb.contains_similar("nonexistent", vec, threshold=0.95)

    def test_contains_similar_empty_predicate(
        self, kb: KnowledgeBase, backend: FHRRBackend
    ) -> None:
        """Test contains_similar returns False when predicate has no facts."""
        vec1 = backend.generate_random(jax.random.PRNGKey(0), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(1), (backend.dimension,))

        # Insert fact then clear predicate
        kb.insert("parent", vec1, ("alice", "bob"))
        kb.clear_predicate("parent")

        # Should return False (no facts to compare against)
        assert not kb.contains_similar("parent", vec2, threshold=0.95)

    def test_contains_similar_different_threshold(
        self, kb: KnowledgeBase, backend: FHRRBackend
    ) -> None:
        """Test contains_similar with different threshold values."""
        # Generate two somewhat similar vectors (same seed + slight variation)
        vec1 = backend.generate_random(jax.random.PRNGKey(42), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(43), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))

        # With very high threshold (0.99), different vectors should not match
        assert not kb.contains_similar("parent", vec2, threshold=0.99)

        # With lower threshold, more likely to match (but still unlikely for random vectors)
        # Just test that the function accepts different thresholds
        kb.contains_similar("parent", vec2, threshold=0.5)

    def test_contains_similar_multiple_facts(self, kb: KnowledgeBase, backend: FHRRBackend) -> None:
        """Test contains_similar checks against all facts in predicate."""
        # Insert multiple facts
        vec1 = backend.generate_random(jax.random.PRNGKey(1), (backend.dimension,))
        vec2 = backend.generate_random(jax.random.PRNGKey(2), (backend.dimension,))
        vec3 = backend.generate_random(jax.random.PRNGKey(3), (backend.dimension,))

        kb.insert("parent", vec1, ("alice", "bob"))
        kb.insert("parent", vec2, ("bob", "carol"))

        # Check if vec3 matches any existing fact
        # Should check all facts, not just first one
        is_similar = kb.contains_similar("parent", vec3, threshold=0.95)

        # Random vectors should not be similar
        assert not is_similar
