"""Test updated persistence layer with shift encoding."""

from pathlib import Path
from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kb.persistence import save_kb, load_kb
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry

print("=== Testing Persistence with Shift Encoding ===\n")

# Setup
backend = FHRRBackend(dim=8192, seed=42)
registry = SymbolRegistry(backend, seed=42)
encoder = VSAEncoder(backend, registry, seed=42)
kb = KnowledgeBase(backend)

# Insert facts
facts = [
    ("parent", ["alice", "bob"]),
    ("parent", ["alice", "carol"]),
    ("parent", ["bob", "dave"]),
]

print("Step 1: Insert facts into KB")
for predicate, args in facts:
    atom_vec = encoder.encode_atom(predicate, args)
    kb.insert(predicate, atom_vec, tuple(args))
    print(f"  Inserted: {predicate}({', '.join(args)})")

print(f"\nKB has {kb.count('parent')} parent facts")

# Save KB
test_path = Path("test_kb.h5")
print(f"\nStep 2: Save KB to {test_path}")
save_kb(kb, test_path)
print(f"  Saved successfully")

# Load KB
print(f"\nStep 3: Load KB from {test_path}")
backend2 = FHRRBackend(dim=8192, seed=42)
kb_loaded = load_kb(backend2, test_path)
print(f"  Loaded successfully")

# Verify
print(f"\nStep 4: Verify loaded KB")
print(f"  Facts count: {kb_loaded.count('parent')}")
print(f"  Facts: {kb_loaded.get_facts('parent')}")
print(f"  Vectors count: {len(kb_loaded.get_vectors('parent'))}")

# Cleanup
test_path.unlink()
print(f"\nCleaned up {test_path}")

# Check
if kb_loaded.count('parent') == 3 and len(kb_loaded.get_vectors('parent')) == 3:
    print("\n*** SUCCESS! Persistence works correctly ***")
    exit(0)
else:
    print("\n*** FAILED ***")
    exit(1)
