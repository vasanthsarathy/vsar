"""Tests for fact store and indexing (Phase 3.2)."""

import pytest
import jax
import jax.numpy as jnp

from vsar.kernel.vsa_backend import FHRRBackend
from vsar.store.fact_store import FactStore
from vsar.store.item import Item, ItemKind, Provenance
from vsar.store.belief import Literal, BeliefState


class TestFactStoreBasics:
    """Test basic fact store operations."""

    def test_create_fact_store(self):
        """Create an empty fact store."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        assert len(store) == 0
        assert store.predicates() == []

    def test_insert_fact(self):
        """Insert a single fact."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec = jax.random.normal(jax.random.PRNGKey(0), (512,), dtype=jnp.complex64)
        item = Item(vec=vec, kind=ItemKind.FACT, weight=1.0)
        literal = Literal("parent", ("alice", "bob"))

        store.insert(item, literal)

        assert len(store) == 1
        assert "parent" in store.predicates()

    def test_insert_only_facts(self):
        """Test that only facts can be inserted."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec = jax.random.normal(jax.random.PRNGKey(0), (512,), dtype=jnp.complex64)
        rule_item = Item(vec=vec, kind=ItemKind.RULE, weight=1.0)
        literal = Literal("parent", ("alice", "bob"))

        with pytest.raises(ValueError):
            store.insert(rule_item, literal)


class TestPredicateIndexing:
    """Test predicate-based indexing."""

    def test_index_facts_by_predicate(self):
        """Retrieve facts with predicate 'parent'."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        # Insert facts with different predicates
        vec1 = jax.random.normal(jax.random.PRNGKey(1), (512,), dtype=jnp.complex64)
        vec2 = jax.random.normal(jax.random.PRNGKey(2), (512,), dtype=jnp.complex64)
        vec3 = jax.random.normal(jax.random.PRNGKey(3), (512,), dtype=jnp.complex64)

        item1 = Item(vec=vec1, kind=ItemKind.FACT, weight=1.0)
        item2 = Item(vec=vec2, kind=ItemKind.FACT, weight=1.0)
        item3 = Item(vec=vec3, kind=ItemKind.FACT, weight=1.0)

        store.insert(item1, Literal("parent", ("alice", "bob")))
        store.insert(item2, Literal("parent", ("alice", "carol")))
        store.insert(item3, Literal("sibling", ("bob", "carol")))

        # Retrieve by predicate
        parent_facts = store.retrieve_by_predicate("parent")
        sibling_facts = store.retrieve_by_predicate("sibling")

        assert len(parent_facts) == 2
        assert len(sibling_facts) == 1

    def test_retrieve_nonexistent_predicate(self):
        """Retrieve facts for predicate that doesn't exist."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        facts = store.retrieve_by_predicate("nonexistent")

        assert facts == []

    def test_predicates_list(self):
        """Get list of all predicates."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec1 = jax.random.normal(jax.random.PRNGKey(1), (512,), dtype=jnp.complex64)
        vec2 = jax.random.normal(jax.random.PRNGKey(2), (512,), dtype=jnp.complex64)

        store.insert(Item(vec=vec1, kind=ItemKind.FACT), Literal("parent", ("a", "b")))
        store.insert(Item(vec=vec2, kind=ItemKind.FACT), Literal("sibling", ("b", "c")))

        preds = store.predicates()

        assert len(preds) == 2
        assert "parent" in preds
        assert "sibling" in preds


class TestSimilarityRetrieval:
    """Test similarity-based retrieval."""

    def test_retrieve_similar_facts(self):
        """Retrieve top-k similar facts."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        # Create similar vectors
        key = jax.random.PRNGKey(0)
        base_vec = jax.random.normal(key, (512,), dtype=jnp.complex64)
        base_vec = base_vec / jnp.linalg.norm(base_vec)

        # Insert facts
        vec1 = base_vec  # Identical
        vec2 = base_vec * 0.9 + jax.random.normal(jax.random.PRNGKey(1), (512,), dtype=jnp.complex64) * 0.1
        vec2 = vec2 / jnp.linalg.norm(vec2)

        store.insert(Item(vec=vec1, kind=ItemKind.FACT), Literal("p", ("a",)))
        store.insert(Item(vec=vec2, kind=ItemKind.FACT), Literal("p", ("b",)))

        # Query with base vector
        results = store.retrieve_similar(base_vec, k=2)

        assert len(results) == 2
        assert all(isinstance(r[0], Item) for r in results)
        assert all(isinstance(r[1], float) for r in results)
        # First result should have higher similarity
        assert results[0][1] >= results[1][1]

    def test_retrieve_with_threshold(self):
        """Filter results by similarity threshold."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        key = jax.random.PRNGKey(0)
        vec1 = jax.random.normal(key, (512,), dtype=jnp.complex64)
        vec1 = vec1 / jnp.linalg.norm(vec1)

        vec2 = jax.random.normal(jax.random.PRNGKey(1), (512,), dtype=jnp.complex64)
        vec2 = vec2 / jnp.linalg.norm(vec2)

        store.insert(Item(vec=vec1, kind=ItemKind.FACT), Literal("p", ("a",)))
        store.insert(Item(vec=vec2, kind=ItemKind.FACT), Literal("p", ("b",)))

        # High threshold - should only get very similar items
        results = store.retrieve_similar(vec1, k=10, threshold=0.95)

        assert len(results) >= 1
        assert results[0][1] >= 0.95

    def test_retrieve_filtered_by_predicate(self):
        """Retrieve similar facts filtered by predicate."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec = jax.random.normal(jax.random.PRNGKey(0), (512,), dtype=jnp.complex64)
        vec = vec / jnp.linalg.norm(vec)

        store.insert(Item(vec=vec, kind=ItemKind.FACT), Literal("parent", ("a", "b")))
        store.insert(Item(vec=vec, kind=ItemKind.FACT), Literal("sibling", ("c", "d")))

        # Search only within "parent" facts
        results = store.retrieve_similar(vec, k=10, predicate="parent")

        assert len(results) == 1
        # Verify it's the parent fact (not sibling)


class TestNoveltyDetection:
    """Test novelty checking."""

    def test_check_novelty_new_fact(self):
        """New fact should be novel."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec1 = jax.random.normal(jax.random.PRNGKey(0), (512,), dtype=jnp.complex64)
        vec1 = vec1 / jnp.linalg.norm(vec1)
        vec2 = jax.random.normal(jax.random.PRNGKey(1), (512,), dtype=jnp.complex64)
        vec2 = vec2 / jnp.linalg.norm(vec2)

        store.insert(Item(vec=vec1, kind=ItemKind.FACT), Literal("p", ("a",)))

        # Different vector should be novel
        is_novel = store.check_novelty(vec2, threshold=0.9)

        assert is_novel

    def test_check_novelty_duplicate(self):
        """Duplicate fact should not be novel."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec = jax.random.normal(jax.random.PRNGKey(0), (512,), dtype=jnp.complex64)
        vec = vec / jnp.linalg.norm(vec)

        store.insert(Item(vec=vec, kind=ItemKind.FACT), Literal("p", ("a",)))

        # Same vector should not be novel
        is_novel = store.check_novelty(vec, threshold=0.95)

        assert not is_novel

    def test_novelty_with_predicate_filter(self):
        """Check novelty within specific predicate."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec = jax.random.normal(jax.random.PRNGKey(0), (512,), dtype=jnp.complex64)
        vec = vec / jnp.linalg.norm(vec)

        store.insert(Item(vec=vec, kind=ItemKind.FACT), Literal("parent", ("a", "b")))

        # Same vector but different predicate - should be novel within "sibling"
        is_novel = store.check_novelty(vec, threshold=0.95, predicate="sibling")

        assert is_novel


class TestParaconsistentTracking:
    """Test paraconsistent belief tracking."""

    def test_update_belief_state_positive(self):
        """Add support for L."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec = jax.random.normal(jax.random.PRNGKey(0), (512,), dtype=jnp.complex64)
        literal = Literal("parent", ("alice", "bob"))

        # Insert positive evidence
        item = Item(vec=vec, kind=ItemKind.FACT, weight=0.7)
        store.insert(item, literal)

        belief = store.get_belief(literal)

        assert belief.supp_pos == 0.7
        assert belief.supp_neg == 0.0
        assert belief.is_consistent()

    def test_update_belief_state_negative(self):
        """Add support for ¬L."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec = jax.random.normal(jax.random.PRNGKey(0), (512,), dtype=jnp.complex64)
        literal = Literal("parent", ("alice", "bob"), negated=True)

        # Insert negative evidence
        item = Item(vec=vec, kind=ItemKind.FACT, weight=0.5)
        store.insert(item, literal)

        belief = store.get_belief(literal)

        assert belief.supp_pos == 0.0
        assert belief.supp_neg == 0.5

    def test_paraconsistent_belief(self):
        """Track supp(L) and supp(¬L) independently."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec1 = jax.random.normal(jax.random.PRNGKey(1), (512,), dtype=jnp.complex64)
        vec2 = jax.random.normal(jax.random.PRNGKey(2), (512,), dtype=jnp.complex64)

        literal_pos = Literal("parent", ("alice", "bob"), negated=False)
        literal_neg = Literal("parent", ("alice", "bob"), negated=True)

        # Add evidence for L
        store.insert(Item(vec=vec1, kind=ItemKind.FACT, weight=0.7), literal_pos)

        # Add evidence for ¬L
        store.insert(Item(vec=vec2, kind=ItemKind.FACT, weight=0.4), literal_neg)

        belief = store.get_belief(literal_pos)

        assert belief.supp_pos == 0.7
        assert belief.supp_neg == 0.4
        assert belief.is_contradictory()

    def test_get_belief_unknown_literal(self):
        """Get belief for literal with no evidence."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        literal = Literal("parent", ("alice", "bob"))
        belief = store.get_belief(literal)

        assert belief.is_unknown()
        assert belief.supp_pos == 0.0
        assert belief.supp_neg == 0.0


class TestFactStoreProperties:
    """Test fact store properties."""

    def test_fact_store_repr(self):
        """Test fact store string representation."""
        backend = FHRRBackend(dim=512, seed=42)
        store = FactStore(backend)

        vec = jax.random.normal(jax.random.PRNGKey(0), (512,), dtype=jnp.complex64)
        store.insert(Item(vec=vec, kind=ItemKind.FACT), Literal("p", ("a",)))

        repr_str = repr(store)

        assert "FactStore" in repr_str
        assert "1 facts" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
