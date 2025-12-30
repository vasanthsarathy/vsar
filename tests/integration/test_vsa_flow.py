"""Integration tests for end-to-end VSA flow."""

import pytest

from vsar.encoding.roles import RoleVectorManager
from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.retrieval.query import Retriever
from vsar.symbols.registry import SymbolRegistry


class TestVSAFlow:
    """End-to-end integration tests for VSA reasoning."""

    @pytest.fixture
    def setup_system(self) -> tuple:
        """Set up complete VSA system."""
        # Create backend
        backend = FHRRBackend(dim=512, seed=42)

        # Create registry
        registry = SymbolRegistry(backend, seed=42)

        # Create encoder
        encoder = VSAEncoder(backend, registry, seed=42)

        # Create KB
        kb = KnowledgeBase(backend)

        # Create retriever
        retriever = Retriever(backend, registry, kb, encoder)

        return backend, registry, encoder, kb, retriever

    def test_simple_parent_query(self, setup_system: tuple) -> None:
        """Test simple parent query: parent(alice, X)."""
        backend, registry, encoder, kb, retriever = setup_system

        # Insert facts
        facts = [
            ("alice", "bob"),
            ("alice", "carol"),
        ]

        for args in facts:
            atom_vec = encoder.encode_atom("parent", list(args))
            kb.insert("parent", atom_vec, args)

        # Query: parent(alice, X)
        results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)

        # Verify results
        assert len(results) > 0

        # Check that bob and carol are in results
        entity_names = [name for name, _ in results]
        assert "bob" in entity_names or "carol" in entity_names

        # Check similarity scores are reasonable
        top_score = results[0][1]
        assert top_score > 0.3  # Approximate retrieval threshold

    def test_reverse_parent_query(self, setup_system: tuple) -> None:
        """Test reverse query: parent(X, carol)."""
        backend, registry, encoder, kb, retriever = setup_system

        # Insert facts
        facts = [
            ("alice", "bob"),
            ("bob", "carol"),
        ]

        for args in facts:
            atom_vec = encoder.encode_atom("parent", list(args))
            kb.insert("parent", atom_vec, args)

        # Query: parent(X, carol)
        results = retriever.retrieve("parent", 1, {"2": "carol"}, k=5)

        # Verify bob is in results
        assert len(results) > 0
        entity_names = [name for name, _ in results]
        assert "bob" in entity_names

    def test_grandparent_chain(self, setup_system: tuple) -> None:
        """Test grandparent relationships through chained queries."""
        backend, registry, encoder, kb, retriever = setup_system

        # Insert facts
        parent_facts = [
            ("alice", "bob"),
            ("bob", "carol"),
            ("carol", "dave"),
        ]

        for args in parent_facts:
            atom_vec = encoder.encode_atom("parent", list(args))
            kb.insert("parent", atom_vec, args)

        # Query 1: parent(alice, X) -> should get bob
        results1 = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)
        assert len(results1) > 0
        assert "bob" in [name for name, _ in results1]

        # Query 2: parent(bob, Y) -> should get carol
        results2 = retriever.retrieve("parent", 2, {"1": "bob"}, k=5)
        assert len(results2) > 0
        assert "carol" in [name for name, _ in results2]

        # This demonstrates we can chain queries to find grandparent
        # (though we'd need separate grandparent facts for direct queries)

    def test_multiple_predicates(self, setup_system: tuple) -> None:
        """Test system with multiple predicates."""
        backend, registry, encoder, kb, retriever = setup_system

        # Insert parent facts
        parent_facts = [
            ("alice", "bob"),
            ("carol", "dave"),
        ]

        for args in parent_facts:
            atom_vec = encoder.encode_atom("parent", list(args))
            kb.insert("parent", atom_vec, args)

        # Insert sibling facts
        sibling_facts = [
            ("bob", "eve"),
            ("eve", "bob"),
        ]

        for args in sibling_facts:
            atom_vec = encoder.encode_atom("sibling", list(args))
            kb.insert("sibling", atom_vec, args)

        # Query parent: parent(alice, X)
        parent_results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)
        assert len(parent_results) > 0
        assert "bob" in [name for name, _ in parent_results]

        # Query sibling: sibling(bob, X)
        sibling_results = retriever.retrieve("sibling", 2, {"1": "bob"}, k=5)
        assert len(sibling_results) > 0
        assert "eve" in [name for name, _ in sibling_results]

    def test_ternary_predicate(self, setup_system: tuple) -> None:
        """Test ternary predicate: gave(alice, bob, book)."""
        backend, registry, encoder, kb, retriever = setup_system

        # Insert facts
        facts = [
            ("alice", "bob", "book"),
            ("alice", "carol", "pen"),
            ("dave", "bob", "book"),
        ]

        for args in facts:
            atom_vec = encoder.encode_atom("gave", list(args))
            kb.insert("gave", atom_vec, args)

        # Query: gave(alice, X, book)
        results = retriever.retrieve("gave", 2, {"1": "alice", "3": "book"}, k=5)

        assert len(results) > 0
        entity_names = [name for name, _ in results]
        assert "bob" in entity_names

    def test_high_noise_scenario(self, setup_system: tuple) -> None:
        """Test retrieval with many facts (high noise)."""
        backend, registry, encoder, kb, retriever = setup_system

        # Insert many facts
        facts = [
            ("alice", "bob"),
            ("alice", "carol"),
            ("alice", "dave"),
            ("eve", "frank"),
            ("frank", "grace"),
            ("grace", "henry"),
        ]

        for args in facts:
            atom_vec = encoder.encode_atom("parent", list(args))
            kb.insert("parent", atom_vec, args)

        # Query: parent(alice, X)
        # Should still retrieve alice's children despite noise
        results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)

        assert len(results) > 0
        entity_names = [name for name, _ in results]

        # At least one of alice's children should be in top results
        alice_children = {"bob", "carol", "dave"}
        assert any(child in entity_names for child in alice_children)

    def test_empty_kb_query(self, setup_system: tuple) -> None:
        """Test querying empty KB raises appropriate error."""
        backend, registry, encoder, kb, retriever = setup_system

        # Query without any facts
        with pytest.raises(ValueError, match="not found in KB"):
            retriever.retrieve("parent", 2, {"1": "alice"}, k=5)

    def test_retrieve_all_variables(self, setup_system: tuple) -> None:
        """Test retrieving all unbound variables."""
        backend, registry, encoder, kb, retriever = setup_system

        # Insert facts
        facts = [
            ("alice", "bob", "book"),
        ]

        for args in facts:
            atom_vec = encoder.encode_atom("gave", list(args))
            kb.insert("gave", atom_vec, args)

        # Query: gave(alice, X, Y) - retrieve both X and Y
        results = retriever.retrieve_all_vars("gave", {"1": "alice"}, k=5)

        assert len(results) == 2  # Two unbound positions
        assert 2 in results
        assert 3 in results

        # Check that bob is in position 2 results
        pos2_names = [name for name, _ in results[2]]
        assert "bob" in pos2_names

        # Check that book is in position 3 results
        pos3_names = [name for name, _ in results[3]]
        assert "book" in pos3_names

    def test_deterministic_retrieval(self, setup_system: tuple) -> None:
        """Test that retrieval is deterministic."""
        backend, registry, encoder, kb, retriever = setup_system

        # Insert facts
        facts = [
            ("alice", "bob"),
        ]

        for args in facts:
            atom_vec = encoder.encode_atom("parent", list(args))
            kb.insert("parent", atom_vec, args)

        # Query twice
        results1 = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)
        results2 = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)

        # Results should be identical
        assert len(results1) == len(results2)
        for (name1, score1), (name2, score2) in zip(results1, results2):
            assert name1 == name2
            assert abs(score1 - score2) < 1e-6  # Floating point tolerance
