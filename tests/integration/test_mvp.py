"""End-to-end integration test for MVP (Phase 4.3)."""

import pytest

from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace
from vsar.encoding.atom_encoder import AtomEncoder
from vsar.encoding.atom_encoder import Atom as EncoderAtom, Constant as EncoderConstant
from vsar.unification.decoder import StructureDecoder, Atom, Constant, Variable
from vsar.store.fact_store import FactStore
from vsar.store.item import Item, ItemKind, Provenance
from vsar.store.belief import Literal
from vsar.reasoning.query_engine import QueryEngine


class TestEndToEndReasoning:
    """Test complete reasoning pipeline."""

    def test_mvp_pipeline(self):
        """
        Complete flow:
        1. Register symbols in typed codebooks
        2. Encode facts with bind
        3. Store facts with metadata
        4. Query facts
        5. Verify decoded results
        """
        # 1. Setup infrastructure
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        # 2. Encode and store facts
        facts = [
            ("parent", "alice", "bob"),
            ("parent", "alice", "carol"),
            ("parent", "bob", "dave"),
            ("parent", "carol", "eve"),
        ]

        for pred, arg1, arg2 in facts:
            atom = EncoderAtom(pred, [EncoderConstant(arg1), EncoderConstant(arg2)])
            vec = encoder.encode_atom(atom)
            item = Item(
                vec=vec,
                kind=ItemKind.FACT,
                weight=1.0,
                provenance=Provenance(source="test")
            )
            store.insert(item, Literal(pred, (arg1, arg2)))

        # Verify facts stored
        assert len(store) == 4
        assert "parent" in store.predicates()

        # 3. Query: parent(alice, ?X)
        query1 = Atom("parent", [Constant("alice"), Variable("X")])
        results1 = engine.answer_query(query1, threshold=0.05)

        assert len(results1) == 2
        answers1 = {r.bindings["X"] for r in results1}
        assert answers1 == {"bob", "carol"}

        # 4. Query: parent(?X, dave)
        query2 = Atom("parent", [Variable("X"), Constant("dave")])
        results2 = engine.answer_query(query2, threshold=0.05)

        assert len(results2) == 1
        assert results2[0].bindings["X"] == "bob"

        # 5. Ground query: parent(carol, eve)
        query3 = Atom("parent", [Constant("carol"), Constant("eve")])
        results3 = engine.answer_query(query3, threshold=0.05)

        assert len(results3) == 1
        assert results3[0].bindings == {}

    def test_paraconsistent_beliefs(self):
        """Test paraconsistent belief tracking in integration."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)

        # Add positive evidence
        vec1 = encoder.encode_atom(EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")]))
        item1 = Item(vec=vec1, kind=ItemKind.FACT, weight=0.8)
        literal_pos = Literal("parent", ("alice", "bob"), negated=False)
        store.insert(item1, literal_pos)

        # Add negative evidence
        vec2 = encoder.encode_atom(EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")]))
        item2 = Item(vec=vec2, kind=ItemKind.FACT, weight=0.3)
        literal_neg = Literal("parent", ("alice", "bob"), negated=True)
        store.insert(item2, literal_neg)

        # Check belief state
        belief = store.get_belief(literal_pos)

        assert belief.supp_pos == 0.8
        assert belief.supp_neg == 0.3
        assert belief.is_contradictory()

    def test_weighted_query_results(self):
        """Test that query results respect fact weights."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)
        store = FactStore(backend)
        engine = QueryEngine(decoder, store)

        # Add facts with different weights
        vec1 = encoder.encode_atom(EncoderAtom("reliable", [EncoderConstant("source1")]))
        store.insert(Item(vec=vec1, kind=ItemKind.FACT, weight=0.9), Literal("reliable", ("source1",)))

        vec2 = encoder.encode_atom(EncoderAtom("reliable", [EncoderConstant("source2")]))
        store.insert(Item(vec=vec2, kind=ItemKind.FACT, weight=0.5), Literal("reliable", ("source2",)))

        # Query
        query = Atom("reliable", [Variable("X")])
        results = engine.answer_query(query, threshold=0.05)

        # Results should be sorted by weight
        assert len(results) == 2
        assert results[0].score == 0.9
        assert results[0].bindings["X"] == "source1"
        assert results[1].score == 0.5
        assert results[1].bindings["X"] == "source2"

    def test_symbol_space_isolation(self):
        """Test that different symbol spaces are isolated."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)

        # Register same name in different spaces
        entity_alice = registry.register(SymbolSpace.ENTITIES, "alice")
        pred_alice = registry.register(SymbolSpace.PREDICATES, "alice")

        # Should be different vectors (random, so ~0.5 similarity expected)
        similarity = backend.similarity(entity_alice, pred_alice)
        assert similarity < 0.7  # Different random vectors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
