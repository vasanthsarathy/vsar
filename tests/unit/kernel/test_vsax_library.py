"""Tests for vsax library VSA operations.

These tests verify that the underlying vsax library is working correctly
for basic VSA operations like bind, unbind, bundle, and similarity.
"""

import jax
import jax.numpy as jnp
import pytest
from vsax import create_fhrr_model, sample_complex_random


class TestVSAXBasics:
    """Test basic vsax operations."""

    @pytest.fixture
    def model(self):
        """Create FHRR model."""
        return create_fhrr_model(dim=8192)

    def test_bind_unbind_roundtrip(self, model):
        """Test that bind/unbind is approximately invertible."""
        # Create two random vectors
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        a = sample_complex_random(dim=model.dimension, n=1, key=key1).squeeze(axis=0)
        b = sample_complex_random(dim=model.dimension, n=1, key=key2).squeeze(axis=0)

        # Bind: c = a ⊗ b
        c = model.opset.bind(a, b)

        # Unbind: c ⊗⁻¹ a should ≈ b
        b_recovered = model.opset.unbind(c, a)

        # Check similarity
        similarity = model.opset.similarity(b, b_recovered)
        print(f"\nBind/unbind roundtrip similarity: {similarity:.4f}")

        # Should have high similarity (>0.8 for FHRR)
        assert similarity > 0.8, f"Expected >0.8, got {similarity:.4f}"

    def test_bundle_unbundle(self, model):
        """Test bundling and retrieving from bundle."""
        # Create three vectors
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        a = model.opset.random(keys[0], shape=(model.dimension,))
        b = model.opset.random(keys[1], shape=(model.dimension,))
        c = model.opset.random(keys[2], shape=(model.dimension,))

        # Bundle: bundle = a + b + c (normalized)
        bundle = model.opset.bundle([a, b, c])

        # Check that each vector has reasonable similarity to bundle
        sim_a = model.opset.similarity(bundle, a)
        sim_b = model.opset.similarity(bundle, b)
        sim_c = model.opset.similarity(bundle, c)

        print(f"\nBundle similarities: a={sim_a:.4f}, b={sim_b:.4f}, c={sim_c:.4f}")

        # Each should be moderately similar (bundle is average-ish)
        assert sim_a > 0.3, f"Expected >0.3, got {sim_a:.4f}"
        assert sim_b > 0.3, f"Expected >0.3, got {sim_b:.4f}"
        assert sim_c > 0.3, f"Expected >0.3, got {sim_c:.4f}"

    def test_exact_match_similarity(self, model):
        """Test that identical vectors have similarity ~1.0."""
        key = jax.random.PRNGKey(42)
        a = model.opset.random(key, shape=(model.dimension,))

        # Self-similarity should be ~1.0
        similarity = model.opset.similarity(a, a)
        print(f"\nSelf-similarity: {similarity:.4f}")

        assert similarity > 0.99, f"Expected >0.99, got {similarity:.4f}"

    def test_random_vectors_low_similarity(self, model):
        """Test that random vectors have low similarity."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        a = sample_complex_random(dim=model.dimension, n=1, key=key1).squeeze(axis=0)
        b = sample_complex_random(dim=model.dimension, n=1, key=key2).squeeze(axis=0)

        similarity = model.opset.similarity(a, b)
        print(f"\nRandom vector similarity: {similarity:.4f}")

        # Should be close to 0.5 (uncorrelated)
        assert 0.4 < similarity < 0.6, f"Expected ~0.5, got {similarity:.4f}"


class TestRoleFillerPattern:
    """Test role-filler binding pattern used in VSAR."""

    @pytest.fixture
    def model(self):
        """Create FHRR model."""
        return create_fhrr_model(dim=8192)

    def test_simple_role_filler(self, model):
        """Test encoding and retrieving simple role-filler: role ⊗ entity."""
        # Create role and entity vectors
        key = jax.random.PRNGKey(42)
        key_role, key_entity = jax.random.split(key)

        role = model.opset.random(key_role, shape=(model.dimension,))
        entity = model.opset.random(key_entity, shape=(model.dimension,))

        # Bind: role_filler = role ⊗ entity
        role_filler = model.opset.bind(role, entity)

        # Unbind: role_filler ⊗⁻¹ role should ≈ entity
        entity_recovered = model.opset.unbind(role_filler, role)

        similarity = model.opset.similarity(entity, entity_recovered)
        print(f"\nRole-filler unbind similarity: {similarity:.4f}")

        assert similarity > 0.8, f"Expected >0.8, got {similarity:.4f}"

    def test_atom_encoding_pattern(self, model):
        """Test full atom encoding: predicate ⊗ (role1 ⊗ arg1 + role2 ⊗ arg2)."""
        # Create vectors for predicate, roles, and entities
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        predicate = model.opset.random(keys[0], shape=(model.dimension,))
        role1 = model.opset.random(keys[1], shape=(model.dimension,))
        role2 = model.opset.random(keys[2], shape=(model.dimension,))
        alice = model.opset.random(keys[3], shape=(model.dimension,))
        bob = model.opset.random(keys[4], shape=(model.dimension,))

        # Encode atom: parent(alice, bob)
        # atom = predicate ⊗ (role1 ⊗ alice + role2 ⊗ bob)
        rf1 = model.opset.bind(role1, alice)
        rf2 = model.opset.bind(role2, bob)
        bundled = model.opset.bundle([rf1, rf2])
        atom = model.opset.bind(predicate, bundled)

        # Unbind predicate: atom ⊗⁻¹ predicate should ≈ (role1 ⊗ alice + role2 ⊗ bob)
        unbind_pred = model.opset.unbind(atom, predicate)

        # Unbind role1: result ⊗⁻¹ role1 should ≈ alice
        alice_recovered = model.opset.unbind(unbind_pred, role1)
        similarity_alice = model.opset.similarity(alice, alice_recovered)

        # Unbind role2: result ⊗⁻¹ role2 should ≈ bob
        bob_recovered = model.opset.unbind(unbind_pred, role2)
        similarity_bob = model.opset.similarity(bob, bob_recovered)

        print(f"\nAtom unbind similarities: alice={similarity_alice:.4f}, bob={similarity_bob:.4f}")

        # Both should have reasonable similarity
        assert similarity_alice > 0.6, f"Alice: expected >0.6, got {similarity_alice:.4f}"
        assert similarity_bob > 0.6, f"Bob: expected >0.6, got {similarity_bob:.4f}"


class TestQueryPattern:
    """Test query pattern used in VSAR."""

    @pytest.fixture
    def model(self):
        """Create FHRR model."""
        return create_fhrr_model(dim=8192)

    def test_query_unbind_pattern(self, model):
        """Test query pattern: unbind partial query from KB bundle."""
        # Setup: Create vectors
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 6)

        predicate = model.opset.random(keys[0], shape=(model.dimension,))
        role1 = model.opset.random(keys[1], shape=(model.dimension,))
        role2 = model.opset.random(keys[2], shape=(model.dimension,))
        alice = model.opset.random(keys[3], shape=(model.dimension,))
        bob = model.opset.random(keys[4], shape=(model.dimension,))
        carol = model.opset.random(keys[5], shape=(model.dimension,))

        # Encode facts: parent(alice, bob) and parent(alice, carol)
        # fact1 = predicate ⊗ (role1 ⊗ alice + role2 ⊗ bob)
        fact1 = model.opset.bind(
            predicate,
            model.opset.bundle([model.opset.bind(role1, alice), model.opset.bind(role2, bob)]),
        )

        # fact2 = predicate ⊗ (role1 ⊗ alice + role2 ⊗ carol)
        fact2 = model.opset.bind(
            predicate,
            model.opset.bundle([model.opset.bind(role1, alice), model.opset.bind(role2, carol)]),
        )

        # KB bundle = fact1 + fact2
        kb_bundle = model.opset.bundle([fact1, fact2])

        # Query: parent(alice, X) - encode only bound arg
        # query = predicate ⊗ (role1 ⊗ alice)
        query = model.opset.bind(predicate, model.opset.bind(role1, alice))

        # Unbind query from KB: kb_bundle ⊗⁻¹ query
        unbind_query = model.opset.unbind(kb_bundle, query)

        # Unbind role2 to get entity: result ⊗⁻¹ role2
        entity_vec = model.opset.unbind(unbind_query, role2)

        # Check similarity to bob and carol
        sim_bob = model.opset.similarity(entity_vec, bob)
        sim_carol = model.opset.similarity(entity_vec, carol)
        sim_alice = model.opset.similarity(entity_vec, alice)  # Should be low

        print(f"\n=== CRITICAL TEST ===")
        print(
            f"Query result similarities: bob={sim_bob:.4f}, carol={sim_carol:.4f}, alice={sim_alice:.4f}"
        )
        print(f"Expected: bob and carol should have high similarity (>0.6)")
        print(f"Actual max: {max(sim_bob, sim_carol):.4f}")

        # Bob and Carol should have higher similarity than Alice
        # THIS IS THE CRITICAL TEST - if this fails, the query pattern doesn't work!
        assert (
            sim_bob > 0.6 or sim_carol > 0.6
        ), f"Expected high similarity to bob or carol, got bob={sim_bob:.4f}, carol={sim_carol:.4f}"
        assert (
            sim_bob > sim_alice + 0.1 or sim_carol > sim_alice + 0.1
        ), f"Expected bob/carol > alice, but got bob={sim_bob:.4f}, carol={sim_carol:.4f}, alice={sim_alice:.4f}"

    def test_multi_fact_bundle_retrieval(self, model):
        """Test retrieving from bundle with multiple facts."""
        # Create vectors
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 7)

        predicate = model.opset.random(keys[0], shape=(model.dimension,))
        role1 = model.opset.random(keys[1], shape=(model.dimension,))
        role2 = model.opset.random(keys[2], shape=(model.dimension,))

        # Entities
        entities = {
            "alice": model.opset.random(keys[3], shape=(model.dimension,)),
            "bob": model.opset.random(keys[4], shape=(model.dimension,)),
            "carol": model.opset.random(keys[5], shape=(model.dimension,)),
            "dave": model.opset.random(keys[6], shape=(model.dimension,)),
        }

        # Create facts: parent(alice, bob), parent(alice, carol), parent(bob, dave)
        facts = []
        fact_pairs = [
            ("alice", "bob"),
            ("alice", "carol"),
            ("bob", "dave"),
        ]

        for arg1_name, arg2_name in fact_pairs:
            arg1 = entities[arg1_name]
            arg2 = entities[arg2_name]
            fact = model.opset.bind(
                predicate,
                model.opset.bundle([model.opset.bind(role1, arg1), model.opset.bind(role2, arg2)]),
            )
            facts.append(fact)

        # Bundle all facts
        kb_bundle = model.opset.bundle(facts)

        # Query: parent(alice, X)
        query = model.opset.bind(predicate, model.opset.bind(role1, entities["alice"]))
        unbind_query = model.opset.unbind(kb_bundle, query)
        entity_vec = model.opset.unbind(unbind_query, role2)

        # Check similarities to all entities
        similarities = {}
        for name, entity in entities.items():
            similarities[name] = model.opset.similarity(entity_vec, entity)

        print(f"\nMulti-fact query similarities: {similarities}")
        print(f"Expected: bob and carol should have high similarity")

        # Bob and Carol should have higher similarity than others
        # This tests if bundling multiple facts preserves query ability
        assert (
            similarities["bob"] > 0.5 or similarities["carol"] > 0.5
        ), f"Expected high similarity to bob or carol: {similarities}"
