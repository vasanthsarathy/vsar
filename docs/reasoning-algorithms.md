# VSAR Reasoning Algorithms

**Version**: 1.0
**Date**: 2025-12-31

## Table of Contents

1. [Overview](#overview)
2. [Query Execution via Resonator Filtering](#query-execution-via-resonator-filtering)
3. [Forward Chaining with Semi-Naive Evaluation](#forward-chaining-with-semi-naive-evaluation)
4. [Beam Search Joins](#beam-search-joins)
5. [Variable Substitution](#variable-substitution)
6. [Cleanup (Symbol Decoding)](#cleanup-symbol-decoding)
7. [Novelty Detection](#novelty-detection)
8. [Complete Algorithm Flows](#complete-algorithm-flows)

---

## Overview

VSAR implements approximate reasoning using Vector Symbolic Architectures (VSAs), specifically the FHRR (Fourier Holographic Reduced Representation) model. This document describes the core algorithms that enable:

- **Retrieval**: Query facts with approximate matching
- **Derivation**: Apply rules to generate new facts
- **Chaining**: Iterative rule application until fixpoint
- **Joining**: Combine multiple query results with bounded search

### Key Design Principles

1. **Shift-based encoding** (not bind/unbind) due to FHRR bind/unbind bugs
2. **Resonator filtering** for weighted retrieval instead of direct unbinding
3. **Beam search** to prevent combinatorial explosion in joins
4. **Semi-naive evaluation** for efficient fixpoint computation
5. **Novelty detection** to avoid duplicate derived facts
6. **Vectorized operations** throughout (JAX-based)

---

## Query Execution via Resonator Filtering

### Algorithm: `Retriever.retrieve(predicate, var_position, bound_args, k)`

**Purpose**: Retrieve top-k values for a variable in a query like `parent(alice, X)?`

**Input**:
- `predicate`: Predicate name (e.g., "parent")
- `var_position`: Position of variable (1-indexed, e.g., 2 for second argument)
- `bound_args`: Dictionary mapping positions to values (e.g., `{"1": "alice"}`)
- `k`: Number of results to return

**Output**: List of `(entity, score)` tuples, sorted by similarity score

### Mathematical Formulation

Given:
- Facts encoded as: `enc(p(t1, t2)) = shift(hv(t1), 1) ⊕ shift(hv(t2), 2)`
- Query: `p(alice, X)?` where we want to find X

**Problem**: We can't directly unbind (due to FHRR bugs), so we use resonator filtering.

### Resonator Filtering Algorithm

**Step 1: Get all fact vectors for predicate**
```
fact_vecs = KB.get_vectors(predicate)
```

**Step 2: Compute weight for each fact (resonator filtering)**

For each fact vector `f_i`:
```
weight_i = 1.0
for each bound position p with value v:
    decoded_p = permute(f_i, -p)        # Decode position p
    entity_vec = hv(v)                   # Get expected value vector
    sim = cosine_similarity(decoded_p, entity_vec)
    weight_i *= max(0, sim)              # Multiply similarities
```

This implements a **soft filter**: facts that match all bound arguments get high weights, others get low weights.

**Step 3: Create weighted bundle**
```
weighted_bundle = Σ(weight_i * f_i) for all i
weighted_bundle = normalize(weighted_bundle)
```

Facts that match the bound arguments contribute more to the bundle.

**Step 4: Decode variable position**
```
entity_vec = permute(weighted_bundle, -var_position)
```

**Step 5: Cleanup to find top-k symbols**
```
results = cleanup(ENTITIES, entity_vec, k)
```

### Special Case: No Bound Arguments

If the query has no bound arguments (e.g., `parent(X, Y)?`):
```
for each fact_vec in fact_vecs:
    decoded = permute(fact_vec, -var_position)
    entity, score = cleanup(ENTITIES, decoded, k=1)
    candidates.append((entity, score))

sort candidates by score
return top k
```

### Example Trace

```
Query: parent(alice, X)?
Facts in KB:
  - parent(alice, bob)     encoded as f1
  - parent(alice, carol)   encoded as f2
  - parent(dave, eve)      encoded as f3

Step 1: Get fact vectors
  fact_vecs = [f1, f2, f3]

Step 2: Compute weights (position 1 = alice)
  For f1:
    decoded_1 = permute(f1, -1)  → approximately hv(alice)
    sim = cos(decoded_1, hv(alice)) = 0.95
    weight_1 = 0.95

  For f2:
    decoded_1 = permute(f2, -1)  → approximately hv(alice)
    sim = cos(decoded_1, hv(alice)) = 0.93
    weight_2 = 0.93

  For f3:
    decoded_1 = permute(f3, -1)  → approximately hv(dave)
    sim = cos(decoded_1, hv(alice)) = 0.05
    weight_3 = 0.05

Step 3: Weighted bundle
  weighted_bundle = 0.95*f1 + 0.93*f2 + 0.05*f3
  weighted_bundle = normalize(weighted_bundle)

Step 4: Decode position 2
  entity_vec = permute(weighted_bundle, -2)
  # This is approximately: 0.95*hv(bob) + 0.93*hv(carol) + 0.05*hv(eve)

Step 5: Cleanup
  results = [
    ("bob", 0.87),
    ("carol", 0.85),
    ("eve", 0.12)
  ]
```

### Complexity

- **Time**: O(m * d) where m = number of facts, d = dimensionality
- **Space**: O(m * d) for fact storage
- **Operations**: All vectorized via JAX

---

## Forward Chaining with Semi-Naive Evaluation

### Algorithm: `apply_rules(engine, rules, max_iterations, k, semi_naive)`

**Purpose**: Apply rules iteratively to derive new facts until fixpoint

**Input**:
- `engine`: VSAR engine with KB
- `rules`: List of Horn clause rules
- `max_iterations`: Maximum iterations (default: 100)
- `k`: Results per query (default: 10)
- `semi_naive`: Use optimization (default: True)

**Output**: `ChainingResult` with statistics

### Semi-Naive Evaluation Algorithm

**Initialization**:
```python
iteration = 0
new_predicates = set(KB.predicates())  # Initially all predicates are "new"
```

**Main Loop**:
```python
while iteration < max_iterations:
    iteration_derived = 0
    next_new_predicates = set()

    for each rule R:
        # Semi-naive optimization: skip rule if no new facts in body
        body_preds = {atom.predicate for atom in R.body}
        if not body_preds.intersect(new_predicates):
            continue  # Skip this rule

        # Track KB state before
        counts_before = {pred: KB.count(pred) for pred in KB.predicates()}

        # Apply rule (see apply_rule algorithm below)
        derived = engine.apply_rule(R, k)
        iteration_derived += derived

        # Track which predicates got new facts
        for pred in KB.predicates():
            if KB.count(pred) > counts_before.get(pred, 0):
                next_new_predicates.add(pred)

    # Update tracking
    new_predicates = next_new_predicates

    # Check fixpoint
    if iteration_derived == 0:
        break  # Fixpoint reached

    iteration += 1

return ChainingResult(...)
```

### Why Semi-Naive?

**Naive evaluation**: Re-apply all rules to all facts every iteration
- **Problem**: Massive redundant work (re-derives same facts)

**Semi-naive evaluation**: Only apply rules when body predicates have new facts
- **Key insight**: A rule can only derive new facts if at least one body predicate has new facts
- **Savings**: Avoids re-evaluation when KB hasn't changed

### Example Trace

```
Base facts:
  parent(alice, bob)
  parent(bob, carol)
  parent(carol, dave)

Rule:
  grandparent(X, Z) :- parent(X, Y), parent(Y, Z)

Iteration 0:
  new_predicates = {parent}

  Apply grandparent rule:
    - body predicates: {parent}
    - parent ∩ new_predicates = {parent}  ✓ Apply rule
    - Derives: grandparent(alice, carol), grandparent(bob, dave)
    - iteration_derived = 2

  next_new_predicates = {grandparent}

Iteration 1:
  new_predicates = {grandparent}

  Apply grandparent rule:
    - body predicates: {parent}
    - parent ∩ new_predicates = {}  ✗ Skip rule (no new parent facts!)
    - iteration_derived = 0

  Fixpoint reached!

Result:
  iterations = 1
  total_derived = 2
  fixpoint_reached = True
```

**Naive version would**:
- Re-apply rule in iteration 1
- Re-derive grandparent(alice, carol), grandparent(bob, dave)
- Detect via novelty check (expensive!)

**Semi-naive version**:
- Skips rule in iteration 1 (no new parent facts)
- No redundant work

### Complexity

- **Naive**: O(i * r * m^b) where i = iterations, r = rules, m = facts, b = body size
- **Semi-naive**: O(i * r * Δm^b) where Δm = new facts per iteration
- **Speedup**: Can be 10-100x for transitive closure

---

## Beam Search Joins

### Algorithm: `join_with_atom(candidates, atom, query_fn, beam_width, k)`

**Purpose**: Join current variable bindings with a new atom, limiting combinatorial explosion

**Input**:
- `candidates`: Current partial bindings (e.g., `[{X: alice, score: 0.9}]`)
- `atom`: Next atom to join (e.g., `parent(X, Y)`)
- `query_fn`: Function to execute queries
- `beam_width`: Maximum candidates to keep (default: 50)
- `k`: Results per query (default: 10)

**Output**: Extended bindings, limited to beam_width

### Algorithm

**For each candidate binding**:
```python
new_candidates = []

for candidate in candidates:
    # Apply current bindings to atom
    partial_atom = candidate.substitution.apply_to_atom(atom)

    # Find unbound variable
    unbound_vars = partial_atom.get_variables()
    if len(unbound_vars) != 1:
        continue  # Skip (multi-variable not supported yet)

    unbound_var = unbound_vars[0]

    # Execute query with partial bindings
    results = execute_query(partial_atom, k)

    # Extend binding with each result
    for value, score in results:
        extended = candidate.extend(unbound_var, value, score)
        new_candidates.append(extended)

# Beam search: keep only top beam_width
new_candidates.sort(key=lambda c: c.score, reverse=True)
return new_candidates[:beam_width]
```

### Beam Search Pruning

**Without beam search**: Combinatorial explosion
```
Rule: ancestor(X, Z) :- parent(X, Y), parent(Y, Z)

If parent has 100 facts:
  - First atom: 100 bindings for Y
  - Second atom: 100 bindings for Z per Y
  - Total: 100 * 100 = 10,000 candidates
```

**With beam_width=50**:
```
  - First atom: 100 bindings, keep top 50
  - Second atom: 50 * 100 = 5,000, keep top 50
  - Total: 50 candidates (200x reduction!)
```

### Score Propagation

Scores are multiplied (approximate joint probability):
```python
def extend(self, var, value, score):
    new_score = self.score * score  # Joint probability
    return CandidateBinding(new_sub, new_score)
```

**Example**:
```
Initial: {X: alice, score: 0.9}
Join with parent(X, Y)? → [(bob, 0.8), (carol, 0.7)]

Results:
  {X: alice, Y: bob, score: 0.9 * 0.8 = 0.72}
  {X: alice, Y: carol, score: 0.9 * 0.7 = 0.63}
```

### Example: Multi-Body Rule

```
Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)

Step 1: Initial candidates from parent(X, Y)
  Execute: parent(X, Y)?
  Results:
    {X: alice, Y: bob, score: 1.0}
    {X: alice, Y: carol, score: 1.0}
    {X: bob, Y: dave, score: 1.0}
    ... (up to k results)

Step 2: Join with parent(Y, Z)
  For {X: alice, Y: bob, score: 1.0}:
    Execute: parent(bob, Z)?
    Results: [(carol, 0.95), (dave, 0.12)]
    Extend:
      {X: alice, Y: bob, Z: carol, score: 0.95}
      {X: alice, Y: bob, Z: dave, score: 0.12}

  For {X: alice, Y: carol, score: 1.0}:
    Execute: parent(carol, Z)?
    Results: [(eve, 0.88)]
    Extend:
      {X: alice, Y: carol, Z: eve, score: 0.88}

  ... (continue for all candidates)

  All new candidates: ~100-1000 candidates
  After beam pruning: Top 50 by score

Step 3: Apply to head
  For {X: alice, Y: bob, Z: carol, score: 0.95}:
    Head: grandparent(X, Z)
    Ground head: grandparent(alice, carol)
    Insert if novel
```

### Complexity

- **Time**: O(c * k * d) per join, where c = candidates, k = results, d = dimensions
- **Space**: O(beam_width) for candidates
- **Joins**: O(|body| - 1) joins per rule

---

## Variable Substitution

### Data Structure: `Substitution`

**Purpose**: Immutable mapping from variables to values

```python
class Substitution:
    bindings: dict[str, str]  # X → alice, Y → bob

    def bind(var, value) → new Substitution
    def get(var) → value | None
    def has(var) → bool
    def apply_to_atom(atom) → ground/partial atom
```

### Algorithm: `apply_to_atom(atom)`

**Purpose**: Replace variables in an atom with their bound values

```python
def apply_to_atom(self, atom: Atom) -> Atom:
    new_args = []
    for arg in atom.args:
        if is_variable(arg) and arg in self.bindings:
            new_args.append(self.bindings[arg])  # Replace with value
        else:
            new_args.append(arg)  # Keep as-is

    return Atom(predicate=atom.predicate, args=new_args)
```

### Example

```python
sub = Substitution(bindings={"X": "alice", "Y": "bob"})

atom1 = Atom("parent", ["X", "Y"])
result1 = sub.apply_to_atom(atom1)
# result1 = Atom("parent", ["alice", "bob"])  ← fully ground

atom2 = Atom("grandparent", ["X", "Z"])
result2 = sub.apply_to_atom(atom2)
# result2 = Atom("grandparent", ["alice", "Z"])  ← partial (Z unbound)
```

### Composition

```python
sub1 = Substitution(bindings={"X": "alice"})
sub2 = Substitution(bindings={"Y": "bob"})

sub3 = sub1.compose(sub2)
# sub3 = Substitution(bindings={"X": "alice", "Y": "bob"})
```

### Variable Detection

```python
def is_variable(term: str) -> bool:
    return len(term) > 0 and term[0].isupper()

# Examples:
is_variable("X")      # True
is_variable("alice")  # False
is_variable("Parent") # True (uppercase, treated as variable!)
```

**Design choice**: Uppercase = variable, lowercase = constant

---

## Cleanup (Symbol Decoding)

### Algorithm: `cleanup(space, vector, registry, backend, k)`

**Purpose**: Decode a noisy hypervector back to symbolic names via nearest neighbor search

**Input**:
- `space`: Symbol space to search (ENTITIES, RELATIONS, etc.)
- `vector`: Noisy hypervector (e.g., decoded from retrieval)
- `registry`: Symbol registry with basis vectors
- `backend`: Kernel backend for similarity
- `k`: Number of top results

**Output**: List of `(symbol_name, similarity_score)` sorted by score

### Implementation

```python
def cleanup(space, vector, registry, backend, k):
    # Delegate to registry
    return registry.cleanup(space, vector, k)
```

### Registry Cleanup

```python
class SymbolRegistry:
    def cleanup(self, space: SymbolSpace, vector, k: int):
        # Get all basis vectors for this space
        basis_vecs = self._basis[space]  # Dict[str, ndarray]

        if not basis_vecs:
            return []

        # Compute similarity to each basis vector
        results = []
        for name, basis_vec in basis_vecs.items():
            sim = backend.similarity(vector, basis_vec)
            results.append((name, float(sim)))

        # Sort by similarity, return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
```

### Vectorized Implementation (Optimized)

```python
def cleanup_vectorized(space, vector, registry, backend, k):
    # Stack all basis vectors into matrix
    names = list(basis_vecs.keys())
    basis_matrix = jnp.stack([basis_vecs[n] for n in names])

    # Batch similarity computation (MUCH faster!)
    similarities = backend.batch_similarity(vector, basis_matrix)

    # Sort and return top k
    top_k_indices = jnp.argsort(similarities)[::-1][:k]
    results = [(names[i], similarities[i]) for i in top_k_indices]
    return results
```

### Example

```
Noisy vector: v = 0.6*hv(bob) + 0.3*hv(alice) + 0.1*hv(carol) + noise

Cleanup in ENTITIES space:
  sim(v, hv(alice)) = 0.35
  sim(v, hv(bob))   = 0.82
  sim(v, hv(carol)) = 0.28
  sim(v, hv(dave))  = 0.05

Results (k=3):
  [("bob", 0.82), ("alice", 0.35), ("carol", 0.28)]
```

### Why It Works

Hypervector properties:
- **Dissimilarity**: Random basis vectors are approximately orthogonal
  ```
  cos(hv(alice), hv(bob)) ≈ 0
  ```

- **Bundling preserves similarity**:
  ```
  cos(hv(bob) ⊕ hv(alice), hv(bob)) > 0  ← bob still detectable
  ```

- **Weighted bundles**:
  ```
  If v = α*hv(bob) + β*hv(alice) and α > β
  Then cos(v, hv(bob)) > cos(v, hv(alice))  ← bob has highest similarity
  ```

### Complexity

- **Naive**: O(n * d) where n = symbols in space, d = dimensionality
- **Vectorized**: O(d) with batch operations (GPU-accelerated)

---

## Novelty Detection

### Algorithm: `KB.contains_similar(predicate, atom_vec, threshold)`

**Purpose**: Check if a fact already exists (approximately) to avoid duplicates

**Input**:
- `predicate`: Predicate name
- `atom_vec`: Encoded atom vector
- `threshold`: Similarity threshold (default: 0.95)

**Output**: True if similar fact exists

### Implementation

```python
def contains_similar(self, predicate: str, atom_vec, threshold: float):
    if predicate not in self._bundles:
        return False

    # Get all fact vectors for this predicate
    fact_vecs = self._vectors[predicate]

    # Check similarity to each fact
    for fact_vec in fact_vecs:
        sim = self.backend.similarity(atom_vec, fact_vec)
        if sim >= threshold:
            return True  # Found similar fact

    return False  # No similar fact
```

### Why Novelty Detection?

**Problem**: Forward chaining can derive the same fact multiple times

```
Facts: parent(alice, bob), parent(bob, carol)
Rule: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)

Iteration 1:
  Derives: ancestor(alice, bob), ancestor(bob, carol)

Iteration 2:
  Query ancestor(bob, Y)? → carol
  Derives: ancestor(alice, carol)  ← NEW

Iteration 3:
  Query ancestor(carol, Y)? → (none)
  Query ancestor(alice, Y)? → bob, carol
  Would derive: ancestor(alice, bob)  ← DUPLICATE!
```

**Solution**: Check similarity before inserting
```python
if not KB.contains_similar("ancestor", enc(alice, bob), threshold=0.95):
    KB.insert("ancestor", enc(alice, bob), ("alice", "bob"))
else:
    # Skip - already exists
```

### Threshold Selection

- **High threshold (0.95-0.99)**: Strict novelty (only skip exact duplicates)
- **Medium threshold (0.80-0.95)**: Approximate novelty (skip similar facts)
- **Low threshold (< 0.80)**: Aggressive deduplication (may skip valid facts)

**Default**: 0.95 (only skip near-exact matches)

### Vectorized Optimization

```python
def contains_similar_fast(self, predicate, atom_vec, threshold):
    if predicate not in self._bundles:
        return False

    # Stack all fact vectors
    fact_matrix = jnp.stack(self._vectors[predicate])

    # Batch similarity
    similarities = self.backend.batch_similarity(atom_vec, fact_matrix)

    # Check if any exceed threshold
    return jnp.any(similarities >= threshold)
```

### Complexity

- **Time**: O(m * d) where m = facts for predicate, d = dimensions
- **Space**: O(1) (just similarity computation)
- **Amortized**: Can be optimized with approximate nearest neighbor (ANN) indices

---

## Complete Algorithm Flows

### Flow 1: Simple Query Execution

```
User query: parent(alice, X)?

1. VSAREngine.query(Query("parent", ["alice", None]))
   ↓
2. Retriever.retrieve("parent", var_position=2, bound_args={"1": "alice"}, k=10)
   ↓
3. Get fact vectors from KB
   fact_vecs = KB.get_vectors("parent")
   ↓
4. Resonator filtering
   For each fact:
     - Decode position 1
     - Compare to hv(alice)
     - Compute weight
   ↓
5. Create weighted bundle
   weighted_bundle = Σ(weight_i * fact_vec_i)
   ↓
6. Decode variable position
   entity_vec = permute(weighted_bundle, -2)
   ↓
7. Cleanup to symbols
   results = cleanup(ENTITIES, entity_vec, k=10)
   ↓
8. Return QueryResult
   results = [("bob", 0.85), ("carol", 0.72), ...]
```

### Flow 2: Query with Rules (Forward Chaining)

```
User query: grandparent(alice, X)?
Rules: [grandparent(X, Z) :- parent(X, Y), parent(Y, Z)]

1. VSAREngine.query(query, rules=[rule])
   ↓
2. apply_rules(engine, [rule], max_iterations=100)
   ↓
3. Iteration 0:
   ├─ For rule: grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
   │  ├─ initial_candidates_from_atom(parent(X,Y))
   │  │  ├─ Execute query: parent(X,Y)?
   │  │  └─ Returns: [{X: alice, Y: bob}, {X: bob, Y: carol}, ...]
   │  ├─ join_with_atom(candidates, parent(Y,Z))
   │  │  ├─ For {X: alice, Y: bob}:
   │  │  │  ├─ Execute: parent(bob, Z)?
   │  │  │  └─ Returns: [{X: alice, Y: bob, Z: carol}]
   │  │  └─ Beam search: keep top 50
   │  └─ Apply to head:
   │     ├─ For {X: alice, Y: bob, Z: carol}:
   │     │  ├─ Ground head: grandparent(alice, carol)
   │     │  ├─ Check novelty: contains_similar?
   │     │  └─ Insert fact
   │     └─ derived_count = 2
   └─ iteration_derived = 2
   ↓
4. Iteration 1:
   └─ (No new parent facts → skip rule → fixpoint)
   ↓
5. Execute query on enriched KB
   ├─ Retriever.retrieve("grandparent", 2, {"1": "alice"})
   └─ Returns: [("carol", 0.89), ...]
   ↓
6. Return QueryResult
```

### Flow 3: Rule Application (Multi-Body)

```
Rule: connected(X, Z) :- edge(X, Y), edge(Y, Z)

1. engine.apply_rule(rule, k=10)
   ↓
2. Check body predicates exist
   ├─ edge in KB? ✓
   └─ Proceed
   ↓
3. initial_candidates_from_atom(edge(X, Y))
   ├─ Get first body atom variables: [X, Y]
   ├─ Multiple variables → enumerate facts
   │  ├─ Get all facts: [(a,b), (b,c), (c,d), ...]
   │  └─ Create bindings:
   │     [{X:a, Y:b, score:1.0}, {X:b, Y:c, score:1.0}, ...]
   └─ candidates = [{X:a, Y:b}, {X:b, Y:c}, {X:c, Y:d}, ...]
   ↓
4. join_with_atom(candidates, edge(Y, Z))
   ├─ For {X:a, Y:b, score:1.0}:
   │  ├─ Partial atom: edge(b, Z)  [Z unbound]
   │  ├─ Execute query: edge(b, Z)?
   │  ├─ Results: [(c, 0.95), (d, 0.12)]
   │  └─ Extend:
   │     [{X:a, Y:b, Z:c, score:0.95}, {X:a, Y:b, Z:d, score:0.12}]
   │
   ├─ For {X:b, Y:c, score:1.0}:
   │  ├─ Execute: edge(c, Z)?
   │  ├─ Results: [(d, 0.89)]
   │  └─ Extend: [{X:b, Y:c, Z:d, score:0.89}]
   │
   ├─ ... (continue for all candidates)
   │
   └─ Beam search: sort by score, keep top 50
   ↓
5. Apply substitutions to head
   For each final candidate:
   ├─ Apply to head: connected(X, Z)
   ├─ Check if ground
   ├─ Encode atom
   ├─ Check novelty: contains_similar(threshold=0.95)
   └─ Insert if novel
   ↓
6. Return derived_count
```

---

## Performance Characteristics

### Query Execution

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Get fact vectors | O(m) | m = facts for predicate |
| Resonator filtering | O(m * d) | d = dimensions, vectorized |
| Weighted bundle | O(m * d) | Single pass |
| Decode position | O(d) | Permutation (cheap) |
| Cleanup | O(n * d) | n = symbols in space, vectorized |
| **Total** | **O(m * d + n * d)** | **Linear in facts + symbols** |

### Forward Chaining

| Strategy | Complexity | Notes |
|----------|-----------|-------|
| Naive | O(i * r * m^b) | i = iterations, r = rules, m = facts, b = body size |
| Semi-naive | O(i * r * Δm^b) | Δm = new facts per iteration |
| **Speedup** | **10-100x** | **For transitive closure** |

### Beam Search Joins

| Parameter | Without Beam | With Beam (w=50) |
|-----------|-------------|------------------|
| Candidates after 1 join | k (e.g., 100) | min(k, 50) |
| Candidates after 2 joins | k² (e.g., 10,000) | 50 * k → 50 |
| Candidates after 3 joins | k³ (e.g., 1,000,000) | 50 |
| **Space** | **O(k^b)** | **O(beam_width)** |
| **Reduction** | **-** | **~1000x for 3-body rules** |

### Novelty Detection

| Method | Complexity | Notes |
|--------|-----------|-------|
| Exact match | O(1) | Hash lookup (not implemented) |
| Similarity check | O(m * d) | m = facts for predicate |
| ANN index | O(log m * d) | Future optimization |

---

## Key Insights

### 1. **Why Shift-Based Encoding?**

**Problem**: VSAX's FHRR bind/unbind operations are broken
- `unbind(bind(a, b), b) ≠ a` (should be equal!)

**Solution**: Use circular permutation (shift)
- `permute(permute(v, k), -k) = v` (perfectly invertible!)
- Encodes position via rotation, not binding

**Trade-off**: Can't encode arbitrary role-filler pairs, only positions

### 2. **Why Resonator Filtering?**

**Problem**: Can't directly unbind due to FHRR bugs

**Solution**: Weighted bundle based on bound argument matches
- Facts matching bound args contribute more
- Effectively implements soft filtering in vector space

**Analogy**: Like a neural attention mechanism over fact vectors

### 3. **Why Beam Search?**

**Problem**: Combinatorial explosion in multi-body rules
- k results per atom → k^b total combinations

**Solution**: Prune to top beam_width at each step
- Keeps high-scoring candidates
- Discards low-probability branches early

**Trade-off**: May miss some valid derivations (completeness vs. tractability)

### 4. **Why Semi-Naive Evaluation?**

**Problem**: Naive re-evaluation wasteful
- Re-derives same facts every iteration
- Only stops via novelty detection (expensive!)

**Solution**: Track which predicates changed
- Only apply rules when inputs changed
- Fixpoint reached when no new facts

**Optimization**: Can skip entire rules, not just fact checks

### 5. **Why Novelty Detection?**

**Problem**: Multiple derivation paths to same fact
- Wastes computation
- Inflates KB size

**Solution**: Similarity-based duplicate detection
- Check if similar fact exists before inserting
- Threshold controls strictness (0.95 = near-exact only)

**VSA advantage**: Similarity check is natural (not just exact equality)

---

## Future Optimizations

### 1. **Approximate Nearest Neighbor (ANN) Indices**
- Replace linear cleanup with FAISS/hnswlib
- O(log n) instead of O(n) for large symbol spaces
- Critical for scaling to millions of symbols

### 2. **Incremental KB Maintenance**
- Track which rules might fire on new facts
- Avoid scanning all rules every iteration

### 3. **Query Planning**
- Reorder body atoms for efficiency
- Small predicates first (fewer candidates)
- Most selective filters first

### 4. **Materialized Views**
- Cache common query results
- Invalidate on KB updates

### 5. **Parallel Execution**
- Multiple rules in parallel (independent derivations)
- Batch query execution (vectorize across queries, not just within)

### 6. **Magic Sets Transformation**
- Optimize backward chaining by goal-directed forward chaining
- Reduce search space by filtering with query constants

---

## References

### Papers

1. **FHRR**: Plate, T. A. (1995). "Holographic Reduced Representations"
2. **VSA Cleanup**: Kanerva, P. (2009). "Hyperdimensional Computing"
3. **Semi-Naive Evaluation**: Ullman, J. D. (1989). "Principles of Database Systems"
4. **Resonator Networks**: Eliasmith, C. (2013). "How to Build a Brain"

### Related Work

- **Datalog**: Traditional logic programming with set semantics
- **Prolog**: Logic programming with backtracking search
- **Answer Set Programming**: Stable model semantics with negation
- **Probabilistic Logic**: Markov Logic Networks, ProbLog, etc.

### VSAR Innovations

1. **Vector-based unification** (approximate, not exact)
2. **Resonator filtering** (soft constraints in vector space)
3. **Beam search joins** (bounded search, not exhaustive)
4. **Novelty detection** (similarity-based, not equality-based)
5. **Shift encoding** (workaround for bind/unbind bugs)

---

## Appendix: Code Locations

| Algorithm | File | Function |
|-----------|------|----------|
| Query execution | `src/vsar/retrieval/query.py` | `Retriever.retrieve()` |
| Forward chaining | `src/vsar/semantics/chaining.py` | `apply_rules()` |
| Rule application | `src/vsar/semantics/engine.py` | `VSAREngine.apply_rule()` |
| Beam search join | `src/vsar/semantics/join.py` | `join_with_atom()` |
| Initial candidates | `src/vsar/semantics/join.py` | `initial_candidates_from_atom()` |
| Substitution | `src/vsar/semantics/substitution.py` | `Substitution` class |
| Cleanup | `src/vsar/retrieval/cleanup.py` | `cleanup()` |
| Novelty detection | `src/vsar/kb/store.py` | `KnowledgeBase.contains_similar()` |

---

**End of Document**
