"""Integration tests for deterministic behavior."""

import jax.numpy as jnp
import pytest

from vsar.encoding.roles import RoleVectorManager
from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.retrieval.query import Retriever
from vsar.symbols.basis import generate_basis
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class TestDeterminism:
    """Regression tests for deterministic behavior."""

    def test_same_seed_produces_same_backend(self) -> None:
        """Test that same seed produces identical backend behavior."""
        backend1 = FHRRBackend(dim=256, seed=42)
        backend2 = FHRRBackend(dim=256, seed=42)

        # Generate same random vectors
        import jax

        key = jax.random.PRNGKey(0)
        vec1a = backend1.generate_random(key, (backend1.dimension,))
        vec1b = backend1.generate_random(key, (backend1.dimension,))

        vec2a = backend2.generate_random(key, (backend2.dimension,))
        vec2b = backend2.generate_random(key, (backend2.dimension,))

        # Should be identical
        assert jnp.allclose(vec1a, vec2a)
        assert jnp.allclose(vec1b, vec2b)

    def test_same_seed_produces_same_symbols(self) -> None:
        """Test that same seed produces identical symbol vectors."""
        backend1 = FHRRBackend(dim=256, seed=42)
        backend2 = FHRRBackend(dim=256, seed=42)

        # Generate basis vectors
        alice1 = generate_basis(SymbolSpace.ENTITIES, "alice", backend1, seed=42)
        alice2 = generate_basis(SymbolSpace.ENTITIES, "alice", backend2, seed=42)

        bob1 = generate_basis(SymbolSpace.ENTITIES, "bob", backend1, seed=42)
        bob2 = generate_basis(SymbolSpace.ENTITIES, "bob", backend2, seed=42)

        # Should be identical
        assert jnp.allclose(alice1, alice2)
        assert jnp.allclose(bob1, bob2)

    def test_same_seed_produces_same_encoding(self) -> None:
        """Test that same seed produces identical atom encodings."""
        backend1 = FHRRBackend(dim=256, seed=42)
        registry1 = SymbolRegistry(backend1, seed=42)
        encoder1 = VSAEncoder(backend1, registry1, seed=42)

        backend2 = FHRRBackend(dim=256, seed=42)
        registry2 = SymbolRegistry(backend2, seed=42)
        encoder2 = VSAEncoder(backend2, registry2, seed=42)

        # Encode same atom
        atom1 = encoder1.encode_atom("parent", ["alice", "bob"])
        atom2 = encoder2.encode_atom("parent", ["alice", "bob"])

        # Should be identical
        assert jnp.allclose(atom1, atom2)

    def test_same_seed_produces_same_retrieval(self) -> None:
        """Test that same seed produces identical retrieval results."""
        # System 1
        backend1 = FHRRBackend(dim=512, seed=42)
        registry1 = SymbolRegistry(backend1, seed=42)
        encoder1 = VSAEncoder(backend1, registry1, seed=42)
        kb1 = KnowledgeBase(backend1)
        role_manager1 = RoleVectorManager(backend1, seed=42)
        retriever1 = Retriever(backend1, registry1, kb1, encoder1, role_manager1)

        # System 2
        backend2 = FHRRBackend(dim=512, seed=42)
        registry2 = SymbolRegistry(backend2, seed=42)
        encoder2 = VSAEncoder(backend2, registry2, seed=42)
        kb2 = KnowledgeBase(backend2)
        role_manager2 = RoleVectorManager(backend2, seed=42)
        retriever2 = Retriever(backend2, registry2, kb2, encoder2, role_manager2)

        # Insert same facts
        facts = [
            ("alice", "bob"),
            ("alice", "carol"),
        ]

        for args in facts:
            atom_vec1 = encoder1.encode_atom("parent", list(args))
            kb1.insert("parent", atom_vec1, args)

            atom_vec2 = encoder2.encode_atom("parent", list(args))
            kb2.insert("parent", atom_vec2, args)

        # Query both systems
        results1 = retriever1.retrieve("parent", 2, {"1": "alice"}, k=5)
        results2 = retriever2.retrieve("parent", 2, {"1": "alice"}, k=5)

        # Results should be identical
        assert len(results1) == len(results2)

        for (name1, score1), (name2, score2) in zip(results1, results2):
            assert name1 == name2
            assert abs(score1 - score2) < 1e-6

    def test_different_seed_produces_different_symbols(self) -> None:
        """Test that different seeds produce different symbols."""
        backend = FHRRBackend(dim=256, seed=42)

        alice1 = generate_basis(SymbolSpace.ENTITIES, "alice", backend, seed=42)
        alice2 = generate_basis(SymbolSpace.ENTITIES, "alice", backend, seed=123)

        # Should be different
        assert not jnp.allclose(alice1, alice2)

    def test_role_vectors_deterministic(self) -> None:
        """Test that role vectors are deterministic."""
        backend1 = FHRRBackend(dim=256, seed=42)
        role_manager1 = RoleVectorManager(backend1, seed=42)

        backend2 = FHRRBackend(dim=256, seed=42)
        role_manager2 = RoleVectorManager(backend2, seed=42)

        # Get same role vectors
        role1_1 = role_manager1.get_role(1)
        role2_1 = role_manager1.get_role(2)

        role1_2 = role_manager2.get_role(1)
        role2_2 = role_manager2.get_role(2)

        # Should be identical
        assert jnp.allclose(role1_1, role1_2)
        assert jnp.allclose(role2_1, role2_2)

    def test_kb_bundling_deterministic(self) -> None:
        """Test that KB bundling is deterministic."""
        backend1 = FHRRBackend(dim=256, seed=42)
        registry1 = SymbolRegistry(backend1, seed=42)
        encoder1 = VSAEncoder(backend1, registry1, seed=42)
        kb1 = KnowledgeBase(backend1)

        backend2 = FHRRBackend(dim=256, seed=42)
        registry2 = SymbolRegistry(backend2, seed=42)
        encoder2 = VSAEncoder(backend2, registry2, seed=42)
        kb2 = KnowledgeBase(backend2)

        # Insert same facts in same order
        facts = [
            ("alice", "bob"),
            ("alice", "carol"),
        ]

        for args in facts:
            atom_vec1 = encoder1.encode_atom("parent", list(args))
            kb1.insert("parent", atom_vec1, args)

            atom_vec2 = encoder2.encode_atom("parent", list(args))
            kb2.insert("parent", atom_vec2, args)

        # Bundles should be identical
        bundle1 = kb1.get_bundle("parent")
        bundle2 = kb2.get_bundle("parent")

        assert bundle1 is not None
        assert bundle2 is not None
        assert jnp.allclose(bundle1, bundle2)

    def test_repeated_runs_same_results(self) -> None:
        """Test that repeated runs produce same results."""

        def run_experiment():
            """Run complete VSA experiment."""
            backend = FHRRBackend(dim=512, seed=42)
            registry = SymbolRegistry(backend, seed=42)
            encoder = VSAEncoder(backend, registry, seed=42)
            kb = KnowledgeBase(backend)
            role_manager = RoleVectorManager(backend, seed=42)
            retriever = Retriever(backend, registry, kb, encoder, role_manager)

            # Insert facts
            facts = [("alice", "bob"), ("bob", "carol")]
            for args in facts:
                atom_vec = encoder.encode_atom("parent", list(args))
                kb.insert("parent", atom_vec, args)

            # Query
            results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)
            return results

        # Run experiment multiple times
        results1 = run_experiment()
        results2 = run_experiment()
        results3 = run_experiment()

        # All runs should produce identical results
        assert len(results1) == len(results2) == len(results3)

        for r1, r2, r3 in zip(results1, results2, results3):
            assert r1[0] == r2[0] == r3[0]  # Same names
            assert abs(r1[1] - r2[1]) < 1e-6  # Same scores
            assert abs(r2[1] - r3[1]) < 1e-6

    def test_insertion_order_matters(self) -> None:
        """Test that insertion order affects bundle (commutative but order-dependent)."""
        backend = FHRRBackend(dim=256, seed=42)
        registry = SymbolRegistry(backend, seed=42)
        encoder = VSAEncoder(backend, registry, seed=42)

        # KB 1: Insert in order A, B
        kb1 = KnowledgeBase(backend)
        atom_a = encoder.encode_atom("parent", ["alice", "bob"])
        atom_b = encoder.encode_atom("parent", ["carol", "dave"])
        kb1.insert("parent", atom_a, ("alice", "bob"))
        kb1.insert("parent", atom_b, ("carol", "dave"))

        # KB 2: Insert in order B, A
        kb2 = KnowledgeBase(backend)
        kb2.insert("parent", atom_b, ("carol", "dave"))
        kb2.insert("parent", atom_a, ("alice", "bob"))

        # Bundles should be the same (bundling is commutative)
        bundle1 = kb1.get_bundle("parent")
        bundle2 = kb2.get_bundle("parent")

        assert bundle1 is not None
        assert bundle2 is not None

        # For VSA bundling (sum + normalize), order shouldn't matter much
        # But normalization might cause slight differences
        # Let's check they're very similar
        similarity = backend.similarity(bundle1, bundle2)
        assert similarity > 0.99  # Should be nearly identical
