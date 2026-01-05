"""Tests for query answering (Phase 4.1)."""

import pytest
from vsar.encoding.atom_encoder import Atom as EncoderAtom
from vsar.encoding.atom_encoder import AtomEncoder
from vsar.encoding.atom_encoder import Constant as EncoderConstant
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.reasoning.query_engine import QueryEngine
from vsar.store.belief import Literal
from vsar.store.fact_store import FactStore
from vsar.store.item import Item, ItemKind, Provenance
from vsar.symbols.registry import SymbolRegistry
from vsar.unification.decoder import Atom, Constant, StructureDecoder, Variable


@pytest.mark.xfail(reason="QueryEngine WIP - reasoning module integration pending")
class TestQueryAnswering:
    """Test basic query answering."""

    def test_query_single_variable(self):
        """
        Facts: parent(alice, bob), parent(alice, carol)
        Query: parent(alice, ?X)
        Expected: [bob, carol]
        """
        # Setup
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        # Insert facts
        atom1 = EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        vec1 = encoder.encode_atom(atom1)
        item1 = Item(vec=vec1, kind=ItemKind.FACT, weight=1.0, provenance=Provenance("test"))
        store.insert(item1, Literal("parent", ("alice", "bob")))

        atom2 = EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("carol")])
        vec2 = encoder.encode_atom(atom2)
        item2 = Item(vec=vec2, kind=ItemKind.FACT, weight=1.0, provenance=Provenance("test"))
        store.insert(item2, Literal("parent", ("alice", "carol")))

        # Query: parent(alice, ?X)
        query = Atom("parent", [Constant("alice"), Variable("X")])
        results = engine.answer_query(query, threshold=0.05)

        # Check results
        assert len(results) == 2
        answers = {r.bindings["X"] for r in results}
        assert answers == {"bob", "carol"}
        assert all(r.score == 1.0 for r in results)

    def test_query_first_arg_variable(self):
        """
        Facts: parent(alice, bob), parent(carol, bob)
        Query: parent(?X, bob)
        Expected: [alice, carol]
        """
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        # Insert facts
        vec1 = encoder.encode_atom(
            EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        )
        store.insert(Item(vec=vec1, kind=ItemKind.FACT), Literal("parent", ("alice", "bob")))

        vec2 = encoder.encode_atom(
            EncoderAtom("parent", [EncoderConstant("carol"), EncoderConstant("bob")])
        )
        store.insert(Item(vec=vec2, kind=ItemKind.FACT), Literal("parent", ("carol", "bob")))

        # Query: parent(?X, bob)
        query = Atom("parent", [Variable("X"), Constant("bob")])
        results = engine.answer_query(query, threshold=0.05)

        # Check results
        assert len(results) == 2
        answers = {r.bindings["X"] for r in results}
        assert answers == {"alice", "carol"}

    def test_query_no_matches(self):
        """
        Facts: parent(alice, bob)
        Query: parent(dave, ?X)
        Expected: []
        """
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        # Insert fact
        vec = encoder.encode_atom(
            EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        )
        store.insert(Item(vec=vec, kind=ItemKind.FACT), Literal("parent", ("alice", "bob")))

        # Query: parent(dave, ?X)
        query = Atom("parent", [Constant("dave"), Variable("X")])
        results = engine.answer_query(query, threshold=0.05)

        # Check no results
        assert len(results) == 0

    def test_ground_query_found(self):
        """
        Facts: parent(alice, bob)
        Query: parent(alice, bob)
        Expected: Match (empty bindings)
        """
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        # Insert fact
        vec = encoder.encode_atom(
            EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        )
        store.insert(Item(vec=vec, kind=ItemKind.FACT), Literal("parent", ("alice", "bob")))

        # Query: parent(alice, bob)
        query = Atom("parent", [Constant("alice"), Constant("bob")])
        results = engine.answer_query(query, threshold=0.05)

        # Should find match with empty bindings
        assert len(results) == 1
        assert results[0].bindings == {}
        assert results[0].score == 1.0

    def test_ground_query_not_found(self):
        """
        Facts: parent(alice, bob)
        Query: parent(alice, carol)
        Expected: No match
        """
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        # Insert fact
        vec = encoder.encode_atom(
            EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        )
        store.insert(Item(vec=vec, kind=ItemKind.FACT), Literal("parent", ("alice", "bob")))

        # Query: parent(alice, carol) - different
        query = Atom("parent", [Constant("alice"), Constant("carol")])
        results = engine.answer_query(query, threshold=0.05)

        # No match
        assert len(results) == 0


class TestQueryScoring:
    """Test query result scoring."""

    def test_weighted_facts(self):
        """Test that results inherit fact weights."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        # Insert facts with different weights
        vec1 = encoder.encode_atom(
            EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        )
        store.insert(
            Item(vec=vec1, kind=ItemKind.FACT, weight=0.9), Literal("parent", ("alice", "bob"))
        )

        vec2 = encoder.encode_atom(
            EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("carol")])
        )
        store.insert(
            Item(vec=vec2, kind=ItemKind.FACT, weight=0.6), Literal("parent", ("alice", "carol"))
        )

        # Query
        query = Atom("parent", [Constant("alice"), Variable("X")])
        results = engine.answer_query(query, threshold=0.05)

        # Check weights are preserved
        assert len(results) == 2
        # Results should be sorted by score (descending)
        assert results[0].score == 0.9
        assert results[1].score == 0.6
        assert results[0].bindings["X"] == "bob"
        assert results[1].bindings["X"] == "carol"

    def test_max_results_limit(self):
        """Test limiting number of results."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        # Insert 5 facts
        for i in range(5):
            vec = encoder.encode_atom(
                EncoderAtom("p", [EncoderConstant("a"), EncoderConstant(f"b{i}")])
            )
            store.insert(Item(vec=vec, kind=ItemKind.FACT), Literal("p", ("a", f"b{i}")))

        # Query with max_results=3
        query = Atom("p", [Constant("a"), Variable("X")])
        results = engine.answer_query(query, threshold=0.05, max_results=3)

        assert len(results) <= 3


class TestQueryEdgeCases:
    """Test edge cases."""

    def test_query_nonexistent_predicate(self):
        """Query for predicate with no facts."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        # Insert fact with different predicate
        vec = encoder.encode_atom(
            EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        )
        store.insert(Item(vec=vec, kind=ItemKind.FACT), Literal("parent", ("alice", "bob")))

        # Query nonexistent predicate
        query = Atom("sibling", [Constant("alice"), Variable("X")])
        results = engine.answer_query(query, threshold=0.05)

        assert len(results) == 0

    def test_engine_repr(self):
        """Test query engine string representation."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        repr_str = repr(engine)
        assert "QueryEngine" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
