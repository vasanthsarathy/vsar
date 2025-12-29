# VSAR Architecture

This document describes the architecture of VSAR (VSAX Reasoner), a VSA-grounded reasoning system built on hypervector algebra.

## Overview

VSAR implements approximate unification and retrieval using Vector Symbolic Architectures (VSA). The system is organized into five layers:

```
┌─────────────────────────────────────────────┐
│          Retrieval Layer                     │
│  (unbind, cleanup, top-k query)              │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│          KB Storage Layer                    │
│  (predicate-partitioned bundles, HDF5)       │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│          Encoding Layer                      │
│  (role-filler binding, VSA encoder)          │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│          Symbol Layer                        │
│  (typed spaces, basis generation, registry)  │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│          Kernel Layer                        │
│  (FHRR via VSAX, bind/unbind/bundle)         │
└─────────────────────────────────────────────┘
```

## Layer Details

### 1. Kernel Layer (`vsar.kernel`)

Provides low-level VSA operations via the VSAX library.

**Key Components:**
- `KernelBackend` (abstract): Interface for VSA operations
- `FHRRBackend`: FHRR (Fourier Holographic Reduced Representations) implementation
  - Uses complex-valued vectors
  - Bind: Circular convolution via FFT
  - Unbind: Complex conjugate + normalization
  - Bundle: Element-wise sum + normalization
  - Similarity: Cosine similarity
- `MAPBackend`: MAP (Multiply-Add-Permute) implementation (future)

**Design Decisions:**
- Strategy pattern enables polymorphic backend usage
- All operations return normalized vectors
- Wraps VSAX models for GPU acceleration
- Deterministic generation with JAX PRNGKey

**Code Location:** `src/vsar/kernel/`

### 2. Symbol Layer (`vsar.symbols`)

Manages typed symbol spaces and basis vector generation.

**Key Components:**
- `SymbolSpace` enum: Six typed spaces (E, R, A, C, T, S)
  - **E** (Entities): alice, bob, carol
  - **R** (Relations): parent, sibling
  - **A** (Attributes): color, age
  - **C** (Contexts): historical, hypothetical
  - **T** (Time): timestamps, intervals
  - **S** (Structural): list constructors, tuples

- `SymbolRegistry`: Central registry for all symbols
  - Lazy generation: symbols created on first access
  - Cleanup: Reverse lookup via similarity search
  - HDF5 persistence: Save/load basis vectors

- `generate_basis()`: Deterministic hypervector generation
  - Seed derivation: `hash(space) + hash(name) + seed`
  - Ensures same symbol → same vector across sessions

**Design Decisions:**
- Typed spaces prevent collisions (e.g., entity "parent" vs. relation "parent")
- Deterministic generation enables reproducibility
- Registry pattern centralizes symbol management

**Code Location:** `src/vsar/symbols/`

### 3. Encoding Layer (`vsar.encoding`)

Encodes logical atoms into hypervectors using role-filler binding.

**Key Components:**
- `RoleVectorManager`: Manages role vectors (ρ1, ρ2, ...)
  - Orthogonal roles for different argument positions
  - Deterministic with seed offset (base_seed + 10000 + position)

- `VSAEncoder`: Role-filler binding implementation
  - Formula: `enc(p(t1,...,tk)) = hv(p) ⊗ ((hv(ρ1) ⊗ hv(t1)) ⊕ ... ⊕ (hv(ρk) ⊗ hv(tk)))`
  - Query encoding: Supports None for variables
  - Example: `parent(alice, X)` → `hv(parent) ⊗ (hv(ρ1) ⊗ hv(alice))`

**Design Decisions:**
- Role vectors distinguish argument positions
- Query patterns use None for unbound variables
- Automatic symbol registration in appropriate spaces

**Code Location:** `src/vsar/encoding/`

### 4. KB Storage Layer (`vsar.kb`)

Stores ground atoms as bundled hypervectors, partitioned by predicate.

**Key Components:**
- `KnowledgeBase`: Predicate-partitioned storage
  - Structure: `dict[predicate_name, bundled_hypervector]`
  - Incremental bundling: `new_bundle = old_bundle ⊕ new_atom`
  - Fact lists: `dict[predicate_name, list[fact_tuples]]`

- `save_kb()` / `load_kb()`: HDF5 persistence
  - Bundles stored as datasets
  - Facts stored as JSON attributes
  - Preserves insertion order

**Design Decisions:**
- Predicate partitioning reduces noise during retrieval
- HDF5 format enables efficient large-scale storage
- Maintains both vectors (for retrieval) and facts (for reference)

**Code Location:** `src/vsar/kb/`

### 5. Retrieval Layer (`vsar.retrieval`)

Implements top-k retrieval via unbinding and cleanup.

**Key Components:**
- `unbind_query_from_bundle()`: Extract relevant facts
  - `bundle ⊗^(-1) query → role-filler pairs`

- `unbind_role()`: Isolate entity from role-filler binding
  - `(ρ ⊗ entity) ⊗^(-1) ρ → entity`

- `cleanup()`: Nearest neighbor search
  - Compute similarity against all basis vectors in space
  - Return top-k matches sorted by score

- `Retriever`: High-level query interface
  - `retrieve(predicate, var_position, bound_args, k)` → top-k results
  - Orchestrates: encode query → get bundle → unbind → cleanup

**Retrieval Pipeline:**

```
Query: parent(alice, X)
    ↓
1. Encode query: hv(parent) ⊗ (hv(ρ1) ⊗ hv(alice))
    ↓
2. Get KB bundle: hv(parent(alice,bob)) ⊕ hv(parent(bob,carol)) ⊕ ...
    ↓
3. Unbind query from bundle: bundle ⊗^(-1) query
    → Contains: hv(ρ2 ⊗ bob) + noise
    ↓
4. Unbind role ρ2: result ⊗^(-1) ρ2
    → ~hv(bob) + noise
    ↓
5. Cleanup: find top-k nearest in ENTITIES space
    → [("bob", 0.85), ("carol", 0.42), ...]
```

**Design Decisions:**
- Approximate retrieval: ~50%+ similarity acceptable
- Top-k returns ranked results (not just yes/no)
- Noise from bundling handled via cleanup threshold

**Code Location:** `src/vsar/retrieval/`

## Data Flow Example

Complete example: Insert facts and query.

```python
# 1. Setup system
backend = FHRRBackend(dim=512, seed=42)
registry = SymbolRegistry(backend, seed=42)
encoder = VSAEncoder(backend, registry, seed=42)
kb = KnowledgeBase(backend)
retriever = Retriever(backend, registry, kb, encoder, role_manager)

# 2. Insert fact: parent(alice, bob)
atom_vec = encoder.encode_atom("parent", ["alice", "bob"])
# atom_vec = hv(parent) ⊗ ((hv(ρ1) ⊗ hv(alice)) ⊕ (hv(ρ2) ⊗ hv(bob)))

kb.insert("parent", atom_vec, ("alice", "bob"))
# kb["parent"] = atom_vec (first fact)

# 3. Insert fact: parent(bob, carol)
atom_vec2 = encoder.encode_atom("parent", ["bob", "carol"])
kb.insert("parent", atom_vec2, ("bob", "carol"))
# kb["parent"] = atom_vec ⊕ atom_vec2 (bundled)

# 4. Query: parent(alice, X)
results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)
# [("bob", 0.85), ...]
```

## Determinism and Reproducibility

VSAR ensures deterministic behavior:

1. **Fixed seeds** → identical random vectors
2. **Hash-based** seed derivation → same symbols always get same vectors
3. **Deterministic VSAX** operations (no stochastic elements)
4. **Order-independent** bundling (commutative sum)

Regression tests verify:
- Same seed produces same encodings
- Same seed produces same retrieval results
- Repeated runs produce identical outputs

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Encode atom | O(k·d) | k = arity, d = dimension |
| Insert fact | O(d) | Bundle operation |
| Retrieve | O(n·d + k·log k) | n = symbols in space, k = top-k |
| Cleanup | O(n·d) | Similarity search over n symbols |

### Space Complexity

| Component | Size | Notes |
|-----------|------|-------|
| Hypervector | d floats | Typically d = 256-1024 |
| KB bundle | d floats per predicate | One bundle per predicate |
| Symbol registry | n·d floats | n = total unique symbols |

### Approximate Retrieval

- **Bind/unbind fidelity**: ~50% similarity (acceptable for VSA)
- **Cleanup threshold**: >0.3 similarity typically sufficient
- **Noise tolerance**: Bundling 10-100 facts per predicate still works

## Extension Points

### Future Backend Support

To add Clifford algebra backend:

1. Implement `CliffordBackend(KernelBackend)`
2. Define geometric product for bind
3. Implement reverse/conjugate for unbind
4. Create `CliffordEncoder(AtomEncoder)`

### Scaling Strategies

For large KBs (>1M facts):

1. **Indexing**: Add inverted indices per predicate
2. **Sharding**: Partition KB across multiple bundles
3. **Approximate cleanup**: Use LSH for faster nearest neighbor
4. **Compression**: Store low-precision vectors (float16)

## Testing Strategy

VSAR has 179 tests across three categories:

### Unit Tests (156 tests)
- Kernel operations (bind, unbind, bundle)
- Symbol registration and cleanup
- Encoding (atoms, queries, roles)
- KB storage and persistence
- Retrieval primitives

### Integration Tests (23 tests)
- End-to-end VSA flow
- Persistence workflows (save/load)
- Determinism regression tests

### Coverage Targets
- **Minimum**: 90% coverage (required)
- **Current**: 99.07% coverage
- **Excluded**: Type definitions, version file

## References

- [VSAX Library](https://vsarathy.com/vsax/) - GPU-accelerated VSA operations
- [FHRR Paper](https://www.researchgate.net/publication/2884506_Holographic_Reduced_Representation) - Plate (2003)
- [VSA Survey](https://arxiv.org/abs/2001.11797) - Kleyko et al. (2021)
- [Hyperdimensional Computing](https://redwood.berkeley.edu/wp-content/uploads/2020/08/Kanerva2009.pdf) - Kanerva (2009)
