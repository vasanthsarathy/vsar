"""Compare similarity scores across different dimensions."""

import jax
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kb.store import KnowledgeBase
from vsar.retrieval.query import Retriever


def test_dimension_effect():
    """Test how dimension affects query similarity scores."""

    dimensions = [512, 1024, 2048, 4096, 8192]

    print("\n" + "="*70)
    print("DIMENSION IMPACT ON QUERY SIMILARITY")
    print("="*70)
    print("\nQuery: parent(alice, X)? expecting bob and carol")
    print()

    for dim in dimensions:
        # Setup
        backend = FHRRBackend(dim=dim, seed=42)
        registry = SymbolRegistry(backend, seed=42)
        encoder = VSAEncoder(backend, registry, seed=42)
        kb = KnowledgeBase(backend)
        retriever = Retriever(backend, registry, kb, encoder)

        # Insert facts
        facts = [
            ("alice", "bob"),
            ("alice", "carol"),
            ("bob", "dave"),
            ("carol", "eve"),
        ]

        for parent, child in facts:
            atom_vec = encoder.encode_atom("parent", [parent, child])
            kb.insert("parent", atom_vec, (parent, child))

        # Query
        results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)

        # Get scores for expected results
        bob_score = next((score for name, score in results if name == "bob"), 0.0)
        carol_score = next((score for name, score in results if name == "carol"), 0.0)
        avg_score = (bob_score + carol_score) / 2

        print(f"Dim {dim:5d}: bob={bob_score:.4f}, carol={carol_score:.4f}, avg={avg_score:.4f}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("Higher dimensions = better similarity scores")
    print("For production: recommend 4096 or 8192 dimensions")
    print()


def test_bundling_interference():
    """Show the interference from bundling."""
    print("\n" + "="*70)
    print("BUNDLING INTERFERENCE ANALYSIS")
    print("="*70)

    backend = FHRRBackend(dim=8192, seed=42)
    registry = SymbolRegistry(backend, seed=42)

    # Get entity vectors
    alice = registry.register("ENTITIES", "alice")
    bob = registry.register("ENTITIES", "bob")

    # Encode fact: shift(alice,1) + shift(bob,2)
    shifted_alice = backend.permute(alice, 1)
    shifted_bob = backend.permute(bob, 2)
    fact_vec = backend.normalize(shifted_alice + shifted_bob)

    # Decode position 1
    decoded = backend.permute(fact_vec, -1)

    # Compare with pure alice
    sim_with_alice = backend.similarity(decoded, alice)

    # What we got vs what we wanted
    print(f"\nFact encoding: normalize(shift(alice,1) + shift(bob,2))")
    print(f"Decoded position 1: shift(fact, -1)")
    print(f"\nSimilarity with alice: {sim_with_alice:.4f}")
    print(f"This is < 1.0 because decoded = normalize(alice + shift(bob,1))")
    print(f"\nWith dim=8192, interference is reduced but still present")
    print()


if __name__ == "__main__":
    test_dimension_effect()
    test_bundling_interference()
