"""Integration test for shift-based encoding with resonator filtering."""

from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.retrieval.query import Retriever
from vsar.symbols.registry import SymbolRegistry

# Initialize components
backend = FHRRBackend(dim=8192, seed=42)
registry = SymbolRegistry(backend, seed=42)
encoder = VSAEncoder(backend, registry, seed=42)
kb = KnowledgeBase(backend)
retriever = Retriever(backend, registry, kb, encoder)

print("=== Testing Shift-Based Encoding + Resonator Filtering ===\n")

# Insert facts
facts = [
    ("parent", ["alice", "bob"]),
    ("parent", ["alice", "carol"]),
    ("parent", ["bob", "dave"]),
]

print("Step 1: Insert facts")
for predicate, args in facts:
    atom_vec = encoder.encode_atom(predicate, args)
    kb.insert(predicate, atom_vec, tuple(args))
    print(f"  Inserted: {predicate}({', '.join(args)})")

print(f"\nKB contains {kb.count('parent')} parent facts\n")

# Query: parent(alice, X)?
print("Step 2: Query parent(alice, X)?")
results = retriever.retrieve(
    predicate="parent",
    var_position=2,  # X is at position 2
    bound_args={"1": "alice"},  # alice is at position 1
    k=10
)

print("\nQuery Results:")
for entity, score in results:
    marker = " <-- EXPECTED" if entity in ["bob", "carol"] else ""
    print(f"  {entity:8s}: {score:.4f}{marker}")

# Check results
top2_entities = {results[0][0], results[1][0]}
expected = {"bob", "carol"}

print(f"\nTop-2 results: {sorted(top2_entities)}")
print(f"Expected: {sorted(expected)}")

if top2_entities == expected:
    print("\n*** PERFECT SUCCESS! ***")
    print("Shift encoding + resonator filtering works correctly!")
    exit(0)
elif len(top2_entities & expected) >= 1:
    print(f"\n*** PARTIAL SUCCESS: {len(top2_entities & expected)}/2 ***")
    exit(1)
else:
    print("\n*** FAILED ***")
    exit(1)
