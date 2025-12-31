# Rules & Forward Chaining

This guide covers Horn clause rules and forward chaining in VSAR - the core features of Phase 2.

## Overview

VSAR supports **Horn clause rules** for deductive reasoning. Rules allow you to derive new facts from existing ones through **forward chaining** with **fixpoint detection**.

**Key Concepts:**
- **Horn clauses:** Rules with one head and zero or more body atoms
- **Forward chaining:** Iteratively apply rules until no new facts can be derived
- **Fixpoint:** State where no new facts are derived
- **Semi-naive evaluation:** Optimization that only processes new facts

## Rule Syntax

### Basic Structure

```prolog
rule head(Args) :- body1(Args), body2(Args), ..., bodyN(Args).
```

- **head:** Single atom (the conclusion)
- **body:** Zero or more atoms (the conditions)
- **Variables:** Uppercase (X, Y, Person)
- **Constants:** Lowercase (alice, bob)

### Single-Body Rules

Simplest form - one condition:

```prolog
// If X is a person, then X is a human
rule human(X) :- person(X).

// If X is a parent of Y, then Y is a child of X
rule child(Y, X) :- parent(X, Y).
```

### Multi-Body Rules

Multiple conditions (join):

```prolog
// X is grandparent of Z if X is parent of Y and Y is parent of Z
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

// X and Y are siblings if they share parent Z
rule sibling(X, Y) :- parent(Z, X), parent(Z, Y).
```

### Recursive Rules

Rules that reference themselves (transitive closure):

```prolog
// Base case: parent is ancestor
rule ancestor(X, Y) :- parent(X, Y).

// Recursive case: transitive closure
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

## Forward Chaining

### How It Works

1. **Start with base facts** - Facts explicitly inserted
2. **Apply rules** - For each rule, find variable bindings that satisfy the body
3. **Derive new facts** - Generate head facts from successful bindings
4. **Check novelty** - Only add facts that don't already exist (similarity-based)
5. **Repeat** - Continue until fixpoint (no new facts) or max iterations

### Example Execution

Given:

```prolog
fact parent(alice, bob).
fact parent(bob, carol).

rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

**Iteration 1:**
- Query `parent(X, Y)` → [(alice, bob), (bob, carol)]
- For binding `X=alice, Y=bob`:
  - Query `parent(Y, Z)` with `Y=bob` → [(carol)]
  - Generate `grandparent(alice, carol)` ✓
- For binding `X=bob, Y=carol`:
  - Query `parent(Y, Z)` with `Y=carol` → []
  - No derivation

**Result:** Derived 1 new fact: `grandparent(alice, carol)`

**Iteration 2:**
- No new parent facts → No new derivations
- **Fixpoint reached** ✓

### Configuration

```prolog
@model FHRR(dim=1024, seed=42);         // VSA model configuration
@beam(width=50);                        // Beam width for joins (default: 50)
@novelty(threshold=0.95);               // Novelty threshold (default: 0.95)
```

**Parameters:**
- **beam width:** Controls how many candidate bindings to keep (prevents explosion)
- **novelty threshold:** Similarity threshold for duplicate detection (0.95 = 95% similar)

## Query with Rules

### Automatic Rule Application

```python
from vsar.language.ast import Directive, Fact, Query, Rule, Atom
from vsar.semantics.engine import VSAREngine

# Configure engine
directives = [Directive(name="model", params={"type": "FHRR", "dim": 1024, "seed": 42})]
engine = VSAREngine(directives)

# Insert base facts
engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

# Define rules
rules = [
    Rule(
        head=Atom(predicate="grandparent", args=["X", "Z"]),
        body=[
            Atom(predicate="parent", args=["X", "Y"]),
            Atom(predicate="parent", args=["Y", "Z"]),
        ],
    )
]

# Query with automatic rule application
query = Query(predicate="grandparent", args=["alice", None])
result = engine.query(query, rules=rules, k=10)

# Results include derived facts
for entity, score in result.results:
    print(f"{entity}: {score:.4f}")
```

### Manual Forward Chaining

For more control:

```python
from vsar.semantics.chaining import apply_rules

# Apply rules manually
result = apply_rules(
    engine,
    rules,
    max_iterations=100,    # Stop after 100 iterations
    k=10,                  # Retrieve top-10 per query
    semi_naive=True        # Use semi-naive evaluation (faster)
)

print(f"Iterations: {result.iterations}")
print(f"Total derived: {result.total_derived}")
print(f"Fixpoint reached: {result.fixpoint_reached}")
print(f"Per iteration: {result.derived_per_iteration}")

# Now query the enriched KB
query_result = engine.query(Query(predicate="grandparent", args=["alice", None]))
```

## Semi-Naive Evaluation

### Why It Matters

**Naive evaluation:**
- Re-applies ALL rules to ALL facts every iteration
- Redundant work: most facts don't change

**Semi-naive evaluation:**
- Tracks which predicates got new facts
- Only applies rules when body predicates have new facts
- **Significantly faster** for large KBs

### How It Works

```
Iteration 0 (setup):
  new_predicates = {all predicates in KB}

Iteration 1:
  For each rule:
    If ANY body predicate in new_predicates:
      Apply rule → derive facts
  Track which predicates got new facts → new_predicates

Iteration 2:
  Only apply rules with body predicates in new_predicates
  ...

Fixpoint:
  new_predicates is empty → stop
```

### Configuration

```python
# Enable semi-naive (default)
apply_rules(engine, rules, semi_naive=True)

# Disable for debugging
apply_rules(engine, rules, semi_naive=False)
```

Both produce **identical results**, but semi-naive is faster.

## Novelty Detection

### Purpose

Prevent duplicate derived facts that are too similar to existing facts.

### How It Works

Before inserting a derived fact:
1. Encode fact as hypervector
2. Compare with all existing facts for same predicate
3. If similarity > threshold (default 0.95), skip (duplicate)
4. Otherwise, insert (novel)

### Configuration

```prolog
@novelty(threshold=0.95);   // 95% similarity = duplicate
```

**Higher threshold** (e.g., 0.99):
- Stricter duplicate detection
- More facts stored
- Slower chaining

**Lower threshold** (e.g., 0.90):
- Looser duplicate detection
- Fewer facts stored
- Faster chaining

### Example

```python
from vsar.language.ast import Directive
from vsar.semantics.engine import VSAREngine

# Configure novelty threshold
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 1024, "seed": 42}),
    Directive(name="novelty", params={"threshold": 0.95}),
]
engine = VSAREngine(directives)

# Access threshold
print(engine.novelty_threshold)  # 0.95
```

## Common Patterns

### Pattern 1: Transitive Closure

```prolog
// Reachability in a graph
rule reachable(X, Y) :- edge(X, Y).
rule reachable(X, Z) :- edge(X, Y), reachable(Y, Z).
```

### Pattern 2: Derived Relationships

```prolog
// Grandparent from parent
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

// Uncle/aunt from parent and sibling
rule uncle(X, Z) :- sibling(X, Y), parent(Y, Z).
```

### Pattern 3: Symmetric Relations

```prolog
// Make coauthorship symmetric
rule coauthor(Y, X) :- coauthor(X, Y).

// Make friendship symmetric
rule friend(Y, X) :- friend(X, Y).
```

### Pattern 4: Combining Sources

```prolog
// Unified connection from multiple sources
rule connected(X, Y) :- knows(X, Y).
rule connected(X, Y) :- works_with(X, Y).
rule connected(X, Z) :- connected(X, Y), connected(Y, Z).
```

## Performance Tuning

### Beam Width

Controls candidate bindings in joins:

```prolog
@beam(width=50);    // Default: top-50 candidates
@beam(width=100);   // More candidates, slower but more complete
@beam(width=20);    // Fewer candidates, faster but may miss results
```

**Guidelines:**
- Small KB (<1000 facts): beam=20-50
- Medium KB (1000-10000 facts): beam=50-100
- Large KB (>10000 facts): beam=100-200

### Max Iterations

Prevent infinite loops:

```python
apply_rules(engine, rules, max_iterations=100)  # Stop after 100 iterations
```

**Guidelines:**
- Simple rules: max_iterations=10-20
- Recursive rules: max_iterations=50-100
- Deep transitive closure: max_iterations=100-500

### Novelty Threshold

Balance between speed and completeness:

```prolog
@novelty(threshold=0.99);   // Strict (more facts, slower)
@novelty(threshold=0.95);   // Balanced (default)
@novelty(threshold=0.90);   // Loose (fewer facts, faster)
```

## Tracing

### Enable Tracing

```python
# Query with rules
result = engine.query(query, rules=rules)

# Get trace DAG
trace = engine.trace.get_dag()

# Find chaining events
chaining_events = [e for e in trace if e.type == "chaining"]
for event in chaining_events:
    print(f"Iterations: {event.payload['iterations']}")
    print(f"Total derived: {event.payload['total_derived']}")
    print(f"Fixpoint: {event.payload['fixpoint_reached']}")
    print(f"Per iteration: {event.payload['derived_per_iteration']}")
```

### CLI Tracing

```bash
vsar run program.vsar --trace
```

## Limitations

### Current (v0.3.0)

- **Single-variable queries only** - `parent(alice, ?)` works, `parent(?, ?)` doesn't
- **No negation** - Can't express `not enemy(X, Y)`
- **No aggregation** - Can't count, sum, max
- **Forward chaining only** - No backward chaining

See [Capabilities & Limitations](../capabilities.md) for details.

## Examples

See [Examples](../examples.md) for complete working programs:

- **[Basic Rules](../examples.md#example-1-basic-rules)** - Simple derivation
- **[Family Tree](../examples.md#example-2-family-tree)** - Grandparent inference
- **[Transitive Closure](../examples.md#example-3-transitive-closure)** - Ancestor chains
- **[Org Hierarchy](../examples.md#example-4-organizational-hierarchy)** - Manager chains
- **[Knowledge Graph](../examples.md#example-5-knowledge-graph)** - Multi-relation
- **[Academic Network](../examples.md#example-6-academic-network)** - Complex rules

## Next Steps

- **[Language Reference](../language-reference.md)** - Complete VSARL syntax
- **[Python API](python-api.md)** - Programmatic usage
- **[Performance Tuning](performance.md)** - Optimization tips
- **[Tutorials](../tutorials/rules-and-reasoning.md)** - Step-by-step guides
