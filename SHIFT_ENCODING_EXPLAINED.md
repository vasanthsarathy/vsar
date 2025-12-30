# Shift-Based Encoding Explained

## How It Works with Predicates

### Basic Example: parent(alice, bob)

**Step 1: Get entity vectors**
```
v_alice = random vector for "alice" (from ENTITIES symbol space)
v_bob = random vector for "bob" (from ENTITIES symbol space)
```

**Step 2: Shift by position (1-indexed)**
```
shifted_alice = permute(v_alice, 1)  # Circular shift by 1
shifted_bob = permute(v_bob, 2)      # Circular shift by 2
```

What is `permute(vec, n)`? It's a circular rotation of the vector:
- If vec = [a, b, c, d, e], then permute(vec, 2) = [d, e, a, b, c]
- For complex vectors (FHRR), it rotates in the complex plane
- Key property: permute(permute(v, n), -n) == v (perfectly invertible)

**Step 3: Bundle (sum + normalize)**
```
fact_vec = normalize(shifted_alice + shifted_bob)
```

**Step 4: Store in KB**
```
KB["parent"] = [fact_vec]  # Stored under predicate "parent"
```

**Important:** The predicate name "parent" is NOT encoded in the vector itself!
- Instead, facts are partitioned by predicate in the KB
- KB["parent"] stores all parent facts
- KB["loves"] would store all loves facts separately

### Querying: parent(alice, X)?

**Step 1: Get all parent facts from KB**
```
fact_vecs = KB["parent"]  # List of vectors for parent facts
```

**Step 2: Resonator filtering - find matching facts**

For each fact vector:
```
# Decode position 1 (where alice should be)
decoded_pos1 = permute(fact_vec, -1)  # Inverse shift

# Check if this matches alice
similarity = cosine_similarity(decoded_pos1, v_alice)

# Weight this fact by how well it matches
weight = max(0, similarity)
```

**Step 3: Create weighted bundle of matching facts**
```
weighted_sum = sum(weight_i * fact_vec_i for all facts)
weighted_bundle = normalize(weighted_sum)
```

Facts that match "alice at position 1" get high weight.
Facts that don't match get low/zero weight.

**Step 4: Decode the variable position**
```
# Decode position 2 (where X is)
result_vec = permute(weighted_bundle, -2)
```

**Step 5: Cleanup - find nearest entity**
```
# Compare result_vec to all known entities
similarities = [cosine_similarity(result_vec, v_entity) for all entities]

# Return top-k matches
# Should find "bob" with high similarity
```

## Comparison with Traditional Role-Filler Binding

### Traditional VSA Approach (Bind/Unbind)

**Encoding parent(alice, bob):**
```
# Define role vectors for each position
ROLE_1 = random vector for "first argument"
ROLE_2 = random vector for "second argument"

# Bind entities to roles
bound_alice = bind(ROLE_1, v_alice)  # ROLE_1 * v_alice (FFT convolution)
bound_bob = bind(ROLE_2, v_bob)      # ROLE_2 * v_bob

# Bundle
fact_vec = normalize(bound_alice + bound_bob)
```

**Querying parent(alice, X)?**
```
# Unbind ROLE_1 to check what's bound to it
unbound = unbind(fact_vec, ROLE_1)  # Should recover v_alice

# If matches, unbind ROLE_2 to get answer
result_vec = unbind(fact_vec, ROLE_2)  # Should recover v_bob
```

### Key Differences

| Aspect | Shift-based | Role-filler |
|--------|-------------|-------------|
| **Position encoding** | Implicit (shift amount) | Explicit (role vectors) |
| **Invertibility** | Perfect (shift/unshift) | Approximate (unbind) |
| **Complexity** | Simpler (no role mgmt) | More complex (role vectors) |
| **Query scores** | 0.6710 (better) | 0.6507 (slightly lower) |
| **VSA standard** | Less common | Standard approach |
| **Interpretability** | Position = shift amount | Position = role vector |

## How Predicate Structure is Represented

### What IS encoded in the vector:
- Entity identities (alice, bob)
- Positional information (argument 1, argument 2)

### What is NOT encoded in the vector:
- Predicate name (parent, loves, etc.)
- Predicate arity (2-arity, 3-arity, etc.)

### How predicates are distinguished:

**KB Partitioning:**
```
KB = {
    "parent": [vec1, vec2, vec3, ...],    # All parent facts
    "loves": [vec4, vec5, ...],           # All loves facts
    "grandparent": [vec6, vec7, ...],     # All grandparent facts
}
```

When you query `parent(alice, X)?`:
1. Retrieve facts from `KB["parent"]` only
2. Filter by "alice at position 1"
3. Decode position 2

This works because predicates are never mixed in storage.

## Example with Different Arities

### Binary predicate: parent(alice, bob)
```
fact_vec = normalize(
    permute(v_alice, 1) +
    permute(v_bob, 2)
)
```

### Ternary predicate: gave(alice, bob, book)
```
fact_vec = normalize(
    permute(v_alice, 1) +
    permute(v_bob, 2) +
    permute(v_book, 3)
)
```

Query: `gave(alice, X, book)?` (find what alice gave to whom, knowing the object)
```
# Filter facts by:
# - Position 1 matches alice
# - Position 3 matches book
# Decode position 2 to get bob
```

## Pros and Cons for Overall Project

### Pros of Shift-based (current):
1. **Simpler implementation**
   - No role vector management
   - Fewer moving parts
   - Less code to maintain

2. **Better empirical performance**
   - 3% higher similarity scores (0.6710 vs 0.6507)
   - Already working well

3. **Perfectly invertible**
   - shift(shift(v, n), -n) == v exactly
   - No approximation error in encoding/decoding

4. **Aligns with project goals**
   - CLAUDE.md emphasizes simplicity
   - "Make every change as simple as possible"
   - "Avoid over-engineering"

### Cons of Shift-based (concerns):
1. **Less standard in VSA literature**
   - Most VSA papers use role-filler binding
   - May confuse readers familiar with standard VSA

2. **Less interpretable**
   - Position encoded implicitly by shift amount
   - Role vectors have explicit semantic meaning

3. **May limit future extensions**
   - Named roles (subject, object, instrument) harder to add
   - Complex nested structures may need explicit roles
   - From CLAUDE.md: "Clifford mode when: Order-sensitive structures (sequences, nested terms, paths)"

4. **Theoretical foundation**
   - Role-filler is well-studied in VSA theory
   - Shift-based is less common, fewer theoretical guarantees

## Recommendation

### For Current Phase (MVP):
**KEEP SHIFT-BASED ENCODING**

Reasons:
- Already implemented and working well
- Better performance (3% higher scores)
- Simpler architecture (aligns with project simplicity goals)
- Sufficient for Datalog-style facts and rules
- No immediate benefit from switching

### For Future (Post-MVP):
**Consider hybrid approach when needed:**

1. **Simple relational facts** → Shift-based
   - parent(alice, bob)
   - loves(john, mary)
   - Most Datalog-like facts

2. **Complex structures** → Role-filler binding
   - Nested terms: path([a, b, c])
   - Named roles: action(agent:alice, instrument:hammer, patient:nail)
   - Temporal sequences: before(event1, event2)

3. **Clifford mode for order-sensitive structures**
   - As mentioned in CLAUDE.md roadmap
   - Use geometric algebra when structure matters

### Strategic Question

**Does your project roadmap include:**
- [ ] Nested terms / complex structures?
- [ ] Named semantic roles (agent, patient, instrument)?
- [ ] Integration with ontologies (OWL/DL)?
- [ ] Argumentation / epistemic contexts?

If **YES** → Consider designing for role-filler now (easier migration path)
If **NO** → Shift-based is optimal (simpler, better performance)

Based on CLAUDE.md Phase 3+, you DO plan complex features:
- "Negation + defaults"
- "Argumentation + epistemic contexts"
- "DL/OWL compatibility"

**Nuanced recommendation:**
- Keep shift-based for Phase 1-2 (current work)
- Design abstractions to allow role-filler in Phase 3+
- Add encoder interface that can support both approaches
- This gives you best of both worlds: simplicity now, flexibility later
