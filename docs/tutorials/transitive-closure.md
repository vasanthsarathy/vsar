# Tutorial: Transitive Closure

Learn how to use recursive rules for multi-hop reasoning and transitive closure.

## What You'll Learn

- Recursive rule patterns
- Transitive closure
- Fixpoint detection
- Performance considerations

## The Problem

Given direct relationships, find all indirect relationships:

```
alice → bob → carol → dave → eve

Question: Who are alice's descendants (direct + indirect)?
Answer: bob, carol, dave, eve (all reachable through parent chain)
```

## Solution: Recursive Rules

```prolog
@model FHRR(dim=1024, seed=42);
@beam 50;
@novelty 0.95;

// Base facts: Direct parent relationships
fact parent(alice, bob).
fact parent(bob, carol).
fact parent(carol, dave).
fact parent(dave, eve).

// Rule 1 (Base case): Parents are ancestors
rule ancestor(X, Y) :- parent(X, Y).

// Rule 2 (Recursive case): Transitive closure
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

// Query all of alice's descendants
query ancestor(alice, X)?
```

Output:
```
Applied 2 rules in 4 iterations
Derived 10 new facts
Fixpoint reached: true

Query: ancestor(alice, X)
  bob: 0.9234
  carol: 0.8876
  dave: 0.8456
  eve: 0.8123
```

## How It Works

**Iteration 1 (Base case):**
- `ancestor(alice, bob)` from `parent(alice, bob)`
- `ancestor(bob, carol)` from `parent(bob, carol)`
- `ancestor(carol, dave)` from `parent(carol, dave)`
- `ancestor(dave, eve)` from `parent(dave, eve)`
- **Derived: 4 facts**

**Iteration 2 (1-hop):**
- `ancestor(alice, carol)` from `parent(alice, bob)` + `ancestor(bob, carol)`
- `ancestor(bob, dave)` from `parent(bob, carol)` + `ancestor(carol, dave)`
- `ancestor(carol, eve)` from `parent(carol, dave)` + `ancestor(dave, eve)`
- **Derived: 3 facts**

**Iteration 3 (2-hop):**
- `ancestor(alice, dave)` from `parent(alice, bob)` + `ancestor(bob, dave)`
- `ancestor(bob, eve)` from `parent(bob, carol)` + `ancestor(carol, eve)`
- **Derived: 2 facts**

**Iteration 4 (3-hop):**
- `ancestor(alice, eve)` from `parent(alice, bob)` + `ancestor(bob, eve)`
- **Derived: 1 fact**

**Iteration 5:**
- No new facts derived → **Fixpoint reached!**

## Common Patterns

### Pattern 1: Reachability in Graphs

```prolog
fact edge(a, b).
fact edge(b, c).
fact edge(c, d).

rule reachable(X, Y) :- edge(X, Y).
rule reachable(X, Z) :- edge(X, Y), reachable(Y, Z).

query reachable(a, X)?  // All nodes reachable from 'a'
```

### Pattern 2: Organizational Hierarchy

```prolog
fact manages(ceo, vp).
fact manages(vp, director).
fact manages(director, manager).

rule reports_to(X, Y) :- manages(Y, X).
rule reports_to(X, Z) :- manages(Y, X), reports_to(Y, Z).

query reports_to(manager, X)?  // All levels manager reports to
```

### Pattern 3: Part-of Relationships

```prolog
fact part_of(wheel, car).
fact part_of(car, vehicle).
fact part_of(vehicle, fleet).

rule component_of(X, Y) :- part_of(X, Y).
rule component_of(X, Z) :- part_of(X, Y), component_of(Y, Z).

query component_of(wheel, X)?  // What is wheel ultimately part of?
```

## Performance Tips

### 1. Set Max Iterations

Prevent infinite loops:

```python
from vsar.semantics.chaining import apply_rules

result = apply_rules(engine, rules, max_iterations=100)
```

### 2. Use Semi-Naive Evaluation

Automatically enabled (faster):

```python
result = apply_rules(engine, rules, semi_naive=True)  # Default
```

### 3. Adjust Beam Width

For deep transitive closure:

```prolog
@beam 100;  // Increase for deeper chains
```

### 4. Monitor Progress

```python
result = apply_rules(engine, rules, max_iterations=100)
print(f"Iterations: {result.iterations}")
print(f"Derived per iteration: {result.derived_per_iteration}")
print(f"Fixpoint: {result.fixpoint_reached}")
```

## Exercises

1. **Social Network:** Model friend-of-friend with transitive `knows` rules
2. **Supply Chain:** Track part dependencies through multiple levels
3. **Code Dependencies:** Model import/require relationships transitively

## Next Steps

- **[Knowledge Graphs](knowledge-graphs.md)** - Multi-relation reasoning
- **[Advanced Patterns](advanced-patterns.md)** - Complex rule combinations
- **[Performance Guide](../guides/performance.md)** - Optimization tips
