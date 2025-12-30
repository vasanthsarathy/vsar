"""Complete end-to-end test: insert → save → load → query."""

from pathlib import Path
from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kb.persistence import save_kb, load_kb
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.retrieval.query import Retriever
from vsar.symbols.registry import SymbolRegistry

print("=== Complete End-to-End Test ===\n")

# Phase 1: Create and populate KB
print("Phase 1: Create KB and insert facts")
backend = FHRRBackend(dim=8192, seed=42)
registry = SymbolRegistry(backend, seed=42)
encoder = VSAEncoder(backend, registry, seed=42)
kb = KnowledgeBase(backend)

facts = [
    ("parent", ["alice", "bob"]),
    ("parent", ["alice", "carol"]),
    ("parent", ["bob", "dave"]),
    ("parent", ["carol", "eve"]),
]

for predicate, args in facts:
    atom_vec = encoder.encode_atom(predicate, args)
    kb.insert(predicate, atom_vec, tuple(args))
    print(f"  Inserted: {predicate}({', '.join(args)})")

print(f"\nKB has {kb.count('parent')} facts\n")

# Phase 2: Save KB
print("Phase 2: Save KB to disk")
test_path = Path("test_e2e_kb.h5")
save_kb(kb, test_path)
print(f"  Saved to {test_path}\n")

# Phase 3: Load KB (fresh backend/registry)
print("Phase 3: Load KB from disk")
backend2 = FHRRBackend(dim=8192, seed=42)
registry2 = SymbolRegistry(backend2, seed=42)
encoder2 = VSAEncoder(backend2, registry2, seed=42)

# Important: Re-register all entities used in queries
from vsar.symbols.spaces import SymbolSpace
for predicate, args in facts:
    for arg in args:
        registry2.register(SymbolSpace.ENTITIES, arg)

kb_loaded = load_kb(backend2, test_path)
print(f"  Loaded {kb_loaded.count('parent')} facts\n")

# Phase 4: Query on loaded KB
print("Phase 4: Query loaded KB")
retriever = Retriever(backend2, registry2, kb_loaded, encoder2)

# Query: parent(alice, X)?
print("Query: parent(alice, X)?")
results = retriever.retrieve(
    predicate="parent",
    var_position=2,
    bound_args={"1": "alice"},
    k=5
)

print("\nResults:")
for entity, score in results:
    marker = " <-- EXPECTED" if entity in ["bob", "carol"] else ""
    print(f"  {entity:8s}: {score:.4f}{marker}")

# Verify
top2 = {results[0][0], results[1][0]}
expected = {"bob", "carol"}

success = top2 == expected

# Cleanup
test_path.unlink()
print(f"\nCleaned up {test_path}")

# Result
print("\n" + "="*50)
if success:
    print("*** COMPLETE SUCCESS! ***")
    print("Full workflow works: insert -> save -> load -> query")
    exit(0)
else:
    print(f"*** FAILED ***")
    print(f"Expected: {expected}")
    print(f"Got: {top2}")
    exit(1)
