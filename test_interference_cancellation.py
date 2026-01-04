"""Test interference cancellation with direct facts."""

from vsar.encoding.role_filler_encoder import RoleFillerEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.retrieval.query import Retriever
from vsar.symbols.registry import SymbolRegistry


def test_binary_predicate_cancellation():
    """Test interference cancellation on simple binary predicate."""
    print("\n" + "="*80)
    print("TEST: Interference Cancellation on Binary Predicate parent(alice, X)")
    print("="*80)

    backend = FHRRBackend(dim=8192, seed=42)
    registry = SymbolRegistry(dim=8192, seed=42)
    encoder = RoleFillerEncoder(backend, registry, seed=42)
    kb = KnowledgeBase(backend)

    # Insert facts
    facts = [
        ("alice", "bob"),
        ("alice", "carol"),
        ("alice", "dave"),
    ]

    for args in facts:
        atom_vec = encoder.encode_atom("parent", list(args))
        kb.insert("parent", atom_vec, args)

    # Create retriever
    retriever = Retriever(backend, registry, kb, encoder)

    # Query: parent(alice, X)
    print("\nQuery: parent(alice, X)")
    results = retriever.retrieve("parent", 2, {"1": "alice"}, k=10)

    print(f"\nResults (with interference cancellation):")
    for entity, score in results:
        print(f"  {entity:10s}: {score:.6f}")

    if results:
        avg_score = sum(score for _, score in results[:3]) / min(3, len(results))
        print(f"\nAverage top-3 score: {avg_score:.6f}")

        if avg_score > 0.90:
            print("SUCCESS! Interference cancellation working perfectly (>0.90)")
        elif avg_score > 0.70:
            print(f"PARTIAL: Better than baseline (~0.64) but not optimal")
        else:
            print(f"ISSUE: Scores still low, cancellation may not be working")


def test_without_cancellation_simulation():
    """Simulate what scores would be without cancellation."""
    print("\n" + "="*80)
    print("COMPARISON: Decoding WITHOUT Interference Cancellation")
    print("="*80)

    backend = FHRRBackend(dim=8192, seed=42)
    registry = SymbolRegistry(dim=8192, seed=42)
    encoder = RoleFillerEncoder(backend, registry, seed=42)
    kb = KnowledgeBase(backend)

    # Insert one fact
    atom_vec = encoder.encode_atom("parent", ["alice", "bob"])
    kb.insert("parent", atom_vec, ("alice", "bob"))

    # Get the fact vector
    fact_vec = kb.get_vectors("parent")[0]

    # Get predicate vector
    pred_vec = registry.register(SymbolSpace.PREDICATES, "parent")

    # Unbind predicate
    args_bundle = backend.unbind(fact_vec, pred_vec)
    # args_bundle = shift(alice,1) + shift(bob,2)

    print("\nWithout cancellation:")
    # Decode position 2 directly (has interference from position 1)
    entity_vec_noisy = backend.permute(args_bundle, -2)
    results_noisy = registry.cleanup(SymbolSpace.ENTITIES, entity_vec_noisy, k=1)
    if results_noisy:
        entity, score = results_noisy[0]
        print(f"  Decoded entity: {entity}, score: {score:.6f}")

    print("\nWith cancellation:")
    # Subtract alice's contribution
    alice_vec = registry.register(SymbolSpace.ENTITIES, "alice")
    alice_shifted = backend.permute(alice_vec, 1)
    cleaned_bundle = args_bundle - alice_shifted
    # cleaned_bundle ≈ shift(bob,2)

    # Now decode position 2
    entity_vec_clean = backend.permute(cleaned_bundle, -2)
    results_clean = registry.cleanup(SymbolSpace.ENTITIES, entity_vec_clean, k=1)
    if results_clean:
        entity, score = results_clean[0]
        print(f"  Decoded entity: {entity}, score: {score:.6f}")

    if results_noisy and results_clean:
        improvement = results_clean[0][1] - results_noisy[0][1]
        print(f"\nImprovement: {improvement:.6f} ({improvement/results_noisy[0][1]*100:.1f}%)")


def test_ternary_predicate():
    """Test with ternary predicate to see if cancellation scales."""
    print("\n" + "="*80)
    print("TEST: Ternary Predicate works_at(alice, company, city, X)")
    print("="*80)

    backend = FHRRBackend(dim=8192, seed=42)
    registry = SymbolRegistry(dim=8192, seed=42)
    encoder = RoleFillerEncoder(backend, registry, seed=42)
    kb = KnowledgeBase(backend)

    # Insert facts: works_at(person, company, city)
    facts = [
        ("alice", "acme", "boston"),
        ("alice", "acme", "newyork"),
        ("bob", "techcorp", "seattle"),
    ]

    for args in facts:
        atom_vec = encoder.encode_atom("works_at", list(args))
        kb.insert("works_at", atom_vec, args)

    retriever = Retriever(backend, registry, kb, encoder)

    # Query: works_at(alice, acme, X) - 2 bound args, should remove 2 interference terms
    print("\nQuery: works_at(alice, acme, X)")
    results = retriever.retrieve("works_at", 3, {"1": "alice", "2": "acme"}, k=10)

    print(f"\nResults:")
    for entity, score in results:
        print(f"  {entity:10s}: {score:.6f}")

    if results:
        avg_score = sum(score for _, score in results[:2]) / min(2, len(results))
        print(f"\nAverage score: {avg_score:.6f}")
        print(f"Expected: ~0.95-1.0 with 2 interference terms removed")


if __name__ == "__main__":
    from vsar.symbols.spaces import SymbolSpace

    print("\n" + "="*80)
    print("INTERFERENCE CANCELLATION TESTS")
    print("Expert prediction: scores should jump from ~0.64 to ~0.95-1.0")
    print("="*80)

    test_without_cancellation_simulation()
    test_binary_predicate_cancellation()
    test_ternary_predicate()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("If scores are ~0.95-1.0: Interference cancellation working perfectly")
    print("If scores are ~0.70-0.80: Partial improvement, may need tuning")
    print("If scores are ~0.60-0.65: Cancellation not working, check implementation")
