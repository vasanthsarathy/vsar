# VSA Atom Encoding: Performance Analysis and Cross-Talk Issues

**Date**: 2026-01-02
**Status**: Technical Analysis
**Author**: Implementation Team
**Audience**: VSA/HDC Experts

## Executive Summary

We implemented two atom encoding strategies for VSAR (VSA-grounded reasoning): **shift-based** and **role-filler binding**. Empirical testing reveals that role-filler binding produces significantly lower retrieval scores (~0.31) compared to shift-based encoding (~0.63) due to cross-talk noise in FHRR bind/unbind operations on bundled representations. This document analyzes both approaches and seeks expert guidance on whether this is fundamental to the approach or if there's a better implementation strategy.

---

## 1. Background

### 1.1 System Context

VSAR is a VSA-based reasoning system that encodes logical atoms (predicates with arguments) as hypervectors using FHRR (Fourier Holographic Reduced Representations). Facts are stored in a knowledge base, and queries retrieve entities by unbinding positions and performing cleanup (nearest-neighbor search in symbol space).

**Example**:
```prolog
fact parent(alice, bob).
fact parent(alice, carol).
query parent(alice, X)?  % Should return: bob, carol
```

### 1.2 VSA Backend: FHRR

- **Bind**: Element-wise complex multiplication `a ⊗ b`
- **Unbind**: `unbind(a ⊗ b, b) = a ⊗ b ⊗ b† = a` (b† is complex conjugate)
- **Bundle**: Element-wise addition + normalization `a ⊕ b`
- **Permute**: Circular shift `shift(a, n)`

**Key Properties**:
- Bind/unbind is approximate inverse (not perfect due to numerical precision)
- Permute is **perfectly invertible**: `shift(shift(a, n), -n) = a`

---

## 2. Encoding Approaches

### 2.1 Shift-Based Encoding (Previous Implementation)

**Formula**:
```python
enc(parent(alice, bob)) = shift(hv(alice), 1) ⊕ shift(hv(bob), 2)
```

**Key Characteristics**:
- Each argument shifted by its position index (1-indexed)
- Predicate NOT encoded in the vector (used for KB partitioning only)
- No bind operations - only shift and bundle

**Decoding**:
```python
# To extract position 2:
entity_vec = shift(fact_vec, -2)
# = shift(shift(alice,1) ⊕ shift(bob,2), -2)
# = shift(alice, 1-2) ⊕ shift(bob, 2-2)
# = shift(alice, -1) ⊕ bob
# Cleanup finds: bob (with high similarity)
```

**Properties**:
- ✅ Perfectly invertible positional encoding
- ✅ Minimal cross-talk between positions
- ❌ Predicate not in vector (can't distinguish parent(a,b) from enemy(a,b) by vector alone)

### 2.2 Role-Filler Binding (Current Implementation)

**Formula** (as per design spec):
```python
enc(parent(alice, bob)) = P_parent ⊗ (ARG1 ⊗ hv(alice) ⊕ ARG2 ⊗ hv(bob))
```

Where:
- `P_parent` = predicate vector from PREDICATES symbol space
- `ARG1, ARG2` = role vectors from ARG_ROLES symbol space
- `hv(alice), hv(bob)` = entity vectors from ENTITIES symbol space

**Decoding**:
```python
# To extract position 2:
# Step 1: Unbind predicate
args_bundle = unbind(fact_vec, P_parent)
# = unbind(P_parent ⊗ args_bundle, P_parent)
# = args_bundle  ✓

# Step 2: Unbind role
entity_vec = unbind(args_bundle, ARG2)
# = unbind(ARG1 ⊗ alice ⊕ ARG2 ⊗ bob, ARG2)
# = unbind(ARG1 ⊗ alice, ARG2) ⊕ unbind(ARG2 ⊗ bob, ARG2)
# = (ARG1 ⊗ alice ⊗ ARG2†) ⊕ bob
# = NOISE ⊕ bob  ⚠️
```

**Properties**:
- ✅ Predicate encoded in vector (distinguishes different predicates)
- ✅ Follows VSA role-filler binding conventions
- ❌ **Cross-talk noise** from other role-bound arguments in bundle

---

## 3. Empirical Results

### 3.1 Test Case

From `examples/07_negation.vsar`:
```prolog
fact person(alice).
fact person(bob).
fact person(carol).
fact person(dave).
fact person(eve).

fact enemy(bob, dave).
fact ~enemy(alice, bob).
fact ~enemy(alice, carol).
fact ~enemy(alice, eve).
fact ~enemy(carol, eve).

rule friendly(X, Y) :-
    person(X),
    person(Y),
    not enemy(X, Y),
    not enemy(Y, X).

query friendly(bob, X)?
```

### 3.2 Score Comparison

| Encoding | Query: `friendly(bob, X)` | Top Score | Avg Top-3 |
|----------|---------------------------|-----------|-----------|
| **Shift-based** | eve: 0.6496<br>alice: 0.6419<br>carol: 0.6375 | **0.65** | **0.64** |
| **Role-filler** | eve: 0.3120<br>bob: 0.2787<br>carol: 0.2724 | **0.31** | **0.29** |

**Result**: Role-filler encoding produces **52% lower scores** than shift-based.

### 3.3 Why This Matters

These are **binary predicates** in a reasoning context. Scores below 0.5 are problematic because:
1. Hard to distinguish correct answers from noise
2. Multi-hop reasoning compounds the error (score multiplication)
3. Makes threshold-based filtering unreliable

---

## 4. Root Cause Analysis

### 4.1 The Cross-Talk Problem

When unbinding a role from a bundle of multiple role-bound arguments:

```
unbind(ARG1 ⊗ v1 ⊕ ARG2 ⊗ v2 ⊕ ARG3 ⊗ v3, ARG2)
= unbind(ARG1 ⊗ v1, ARG2) ⊕ unbind(ARG2 ⊗ v2, ARG2) ⊕ unbind(ARG3 ⊗ v3, ARG2)
= (ARG1 ⊗ v1 ⊗ ARG2†) ⊕ v2 ⊕ (ARG3 ⊗ v3 ⊗ ARG2†)
= noise1 ⊕ v2 ⊕ noise3
```

The noise terms `ARGi ⊗ vj ⊗ ARGk†` (where i ≠ k) are approximately random vectors because role vectors are independent random hypervectors.

**Signal-to-noise ratio**:
- Signal: 1 component (the target `v2`)
- Noise: k-1 components (all other role-bound arguments)
- For binary predicate (k=2): SNR ≈ 1:1
- For ternary predicate (k=3): SNR ≈ 1:2
- Gets worse as arity increases

### 4.2 Why Shift-Based Works Better

Shift operation is **linear and invertible**:
```
shift(shift(v1, 1) ⊕ shift(v2, 2), -2)
= shift(v1, 1-2) ⊕ shift(v2, 2-2)
= shift(v1, -1) ⊕ v2
```

The `shift(v1, -1)` term is NOT noise - it's a cleanly separated component. Since we're doing cleanup in ENTITIES space and v1 was shifted, it doesn't match well with entity symbols, effectively filtering it out.

**Key insight**: Shift preserves the **additive structure** of the bundle, while bind/unbind creates **multiplicative cross-talk**.

---

## 5. Implementation Details

### 5.1 Role-Filler Encoder

```python
class RoleFillerEncoder(AtomEncoder):
    def encode_atom(self, predicate: str, args: list[str]) -> jnp.ndarray:
        # Get predicate vector
        pred_vec = self.registry.register(SymbolSpace.PREDICATES, predicate)

        # Encode each argument with its role
        bound_args = []
        for i, arg in enumerate(args):
            entity_vec = self.registry.register(SymbolSpace.ENTITIES, arg)
            role_vec = self._get_role_vector(i + 1)  # ARG1, ARG2, ...
            bound_arg = self.backend.bind(role_vec, entity_vec)
            bound_args.append(bound_arg)

        # Bundle arguments and bind with predicate
        args_bundle = self.backend.bundle(bound_args)
        atom_vec = self.backend.bind(pred_vec, args_bundle)

        return self.backend.normalize(atom_vec)
```

### 5.2 Retriever with Role-Filler

```python
def retrieve(self, predicate: str, var_position: int,
             bound_args: dict[str, str], k: int) -> list[tuple[str, float]]:
    pred_vec = self.registry.register(SymbolSpace.PREDICATES, predicate)

    for fact_vec in self.kb.get_vectors(predicate):
        # Step 1: Unbind predicate
        args_bundle = self.backend.unbind(fact_vec, pred_vec)

        # Step 2: Check bound arguments match
        for pos, entity in bound_args.items():
            role_vec = self._get_role_vector(int(pos))
            decoded = self.backend.unbind(args_bundle, role_vec)
            entity_vec = self.registry.register(SymbolSpace.ENTITIES, entity)

            similarity = self.backend.similarity(decoded, entity_vec)
            if similarity < 0.5:  # Doesn't match
                continue

        # Step 3: Decode variable position
        var_role_vec = self._get_role_vector(var_position)
        entity_vec = self.backend.unbind(args_bundle, var_role_vec)

        # Step 4: Cleanup
        results = self.registry.cleanup(SymbolSpace.ENTITIES, entity_vec, k=1)
        # Results have low scores due to noise!
```

---

## 6. Separate Cleanup Optimization

We also tried **separate cleanup** (decode each fact individually rather than bundling before cleanup):

```python
# Instead of:
weighted_bundle = sum(weight_i * fact_i for i, fact_i in enumerate(facts))
entity_vec = unbind(weighted_bundle, role)
results = cleanup(entity_vec)

# We do:
for fact in facts:
    if matches_bound_args(fact):
        entity_vec = unbind(fact, role)
        result = cleanup(entity_vec)  # Individual cleanup
        candidates.append(result)
```

This helped shift-based encoding (0.37 → 0.63) but did NOT help role-filler (still 0.31) because the fundamental cross-talk issue remains.

---

## 7. Questions for Expert Review

### 7.1 Is This Expected?

**Q1**: Is cross-talk noise in role-filler binding with FHRR an expected and fundamental limitation?

**Q2**: Are there established techniques to mitigate this (e.g., different role generation strategies, interference cancellation)?

### 7.2 Design Specification Alignment

**Q3**: The design document (`docs/encoding-extensibility-design.md`) proposes:
```python
enc(parent(alice, bob)) = ρ1 ⊗ hv(alice) ⊕ ρ2 ⊗ hv(bob)
```

Should this include the predicate? Our interpretation was:
```python
enc(parent(alice, bob)) = P_parent ⊗ (ρ1 ⊗ alice ⊕ ρ2 ⊗ bob)
```

Is this correct, or should predicates be handled differently?

### 7.3 Alternative Approaches

**Q4**: Should we consider:
- **Hybrid encoding**: `α·shift(v, pos) ⊕ β·(role ⊗ v)` for both benefits?
- **Factorization-based decoding**: Use resonator networks instead of direct unbind?
- **Different algebra**: Would MAP-C (Multiply-Add-Permute) perform better?
- **Keep shift-based**: Is there a reason NOT to use shift for positional encoding?

### 7.4 Arity Scaling

**Q5**: How does role-filler binding scale with arity?
- Binary predicates (k=2): 1:1 signal-to-noise
- Ternary (k=3): 1:2 signal-to-noise
- Higher arity: 1:(k-1) signal-to-noise

Is this degradation acceptable, or are there known solutions?

### 7.5 Theoretical vs Practical

**Q6**: In VSA/HDC literature, role-filler binding is widely used. Are there:
- Typical dimensionality requirements for low cross-talk?
- Assumptions about arity or usage patterns?
- Empirical benchmarks we should target?

We're using:
- Dimension: 8192
- Arity: Mostly 2-3
- Model: FHRR (complex-valued)

---

## 8. Recommendations Needed

We seek expert guidance on:

1. **Should we proceed with role-filler encoding despite lower scores?**
   - Is 0.31 similarity acceptable in VSA applications?
   - Will this cause problems in multi-hop reasoning?

2. **What's the "correct" way to implement role-filler binding?**
   - Are we missing a key technique?
   - Should we use different decoding strategies?

3. **Is shift-based encoding acceptable?**
   - Original spec suggested deprecating it
   - But empirically it performs much better
   - Should we keep it as the default?

4. **Path forward?**
   - Implement hybrid approach?
   - Use different VSA model (MAP vs FHRR)?
   - Accept the tradeoff and document it?

---

## 9. References

### Design Documents
- `docs/encoding-extensibility-design.md` - Encoding strategies proposal
- VSAR specification PDFs in `spec/` folder

### Implementation
- `src/vsar/encoding/role_filler_encoder.py` - Role-filler implementation
- `src/vsar/encoding/vsa_encoder.py` - Shift-based implementation (previous)
- `src/vsar/retrieval/query.py` - Retrieval with unbinding

### Test Cases
- `examples/07_negation.vsar` - Negation-as-failure example
- 579 total tests passing with both encoders

### Related Work
- Plate, T. A. (2003). Holographic Reduced Representation
- Kanerva, P. (2009). Hyperdimensional Computing
- Gayler, R. W. (2003). Vector Symbolic Architectures

---

## 10. Appendix: Raw Data

### A. Complete Score Breakdown

**Shift-Based Encoding** (`friendly(bob, X)`):
```
eve    : 0.6496
alice  : 0.6419
carol  : 0.6375
bob    : 0.6270
dave   : 0.6225
```

**Role-Filler Encoding** (`friendly(bob, X)`):
```
eve    : 0.3120
bob    : 0.2787
carol  : 0.2724
alice  : 0.2598
dave   : 0.2426
```

### B. Unary vs Binary Predicate Comparison

| Query Type | Shift-Based | Role-Filler |
|------------|-------------|-------------|
| Unary: `person(X)` | ~1.0 | ~1.0 |
| Binary: `parent(alice, X)` | ~0.63 | ~0.31 |
| Derived: `friendly(bob, X)` | ~0.64 | ~0.31 |

**Observation**: Role-filler degrades significantly for binary predicates, while shift-based maintains high scores.

### C. Vector Dimensionality

All experiments use:
- Dimension: 8192
- Backend: FHRR (complex-valued)
- Seed: 42 (deterministic)
- Normalization: L2 norm

---

**End of Document**

We appreciate any insights or guidance on these encoding strategies. Our goal is to implement the most effective approach for VSA-grounded logical reasoning while maintaining fidelity to established VSA principles.
