# Tutorial: Rules & Reasoning

This tutorial teaches you how to use Horn clause rules to derive new facts through deductive reasoning.

## What You'll Learn

- How to write rules
- Forward chaining basics
- Deriving new relationships
- Query with rules

## Prerequisites

- Completed [Basic Facts & Queries](basic-queries.md)
- Understanding of variables and predicates

## Step 1: Your First Rule

Rules derive new facts from existing ones.

Create `tutorial2.vsar`:

```prolog
@model FHRR(dim=1024, seed=42);
@beam(width=50);

fact person(alice).
fact person(bob).

rule human(X) :- person(X).

query human(X)?
```

Run it:

```bash
vsar run tutorial2.vsar
```

Output:
```
Inserted 2 facts

Applied 1 rule in 1 iteration
Derived 2 new facts
Fixpoint reached: true

┌──────────────────┐
│ Query: human(X)  │
├────────┬─────────┤
│ Entity │ Score   │
├────────┼─────────┤
│ alice  │ 0.9234  │
│ bob    │ 0.9123  │
└────────┴─────────┘
```

**What happened:**
1. Inserted base facts: `person(alice)`, `person(bob)`
2. Applied rule: For each `person(X)`, derive `human(X)`
3. Derived: `human(alice)`, `human(bob)`
4. Query returned derived facts

**Rule syntax:**
```prolog
rule head(Variables) :- body(Variables).
```

## Step 2: Multi-Body Rules (Joins)

Rules can have multiple conditions:

```prolog
@model FHRR(dim=1024, seed=42);
@beam(width=50);

fact parent(alice, bob).
fact parent(bob, carol).

rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

query grandparent(alice, X)?
```

Output:
```
Applied 1 rule in 1 iteration
Derived 1 new fact

Query: grandparent(alice, X)
  carol: 0.8456
```

**How it works:**
1. Find all `parent(X, Y)`: alice→bob, bob→carol
2. For X=alice, Y=bob: Check `parent(bob, Z)` → found carol
3. Derive: `grandparent(alice, carol)`

## Step 3: Recursive Rules

Rules can reference themselves for transitive closure:

```prolog
@model FHRR(dim=1024, seed=42);
@beam(width=50);
@novelty(threshold=0.95);

fact parent(alice, bob).
fact parent(bob, carol).
fact parent(carol, dave).

// Base case: parent is ancestor
rule ancestor(X, Y) :- parent(X, Y).

// Recursive case: transitive closure
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

query ancestor(alice, X)?
```

Output:
```
Applied 2 rules in 3 iterations
Derived 6 new facts
Fixpoint reached: true

Query: ancestor(alice, X)
  bob: 0.9234
  carol: 0.8876
  dave: 0.8123
```

**How it works:**

**Iteration 1:**
- Base rule: Derive `ancestor(alice, bob)`, `ancestor(bob, carol)`, `ancestor(carol, dave)`

**Iteration 2:**
- Recursive rule with alice→bob + bob→carol: Derive `ancestor(alice, carol)`
- Recursive rule with bob→carol + carol→dave: Derive `ancestor(bob, dave)`

**Iteration 3:**
- Recursive rule with alice→bob + bob→dave: Derive `ancestor(alice, dave)`

**Iteration 4:**
- No new facts derived → **Fixpoint!**

## Step 4: Multiple Rules

You can define multiple rules that interact:

```prolog
@model FHRR(dim=1024, seed=42);
@beam(width=50);

fact parent(alice, bob).
fact parent(alice, carol).
fact parent(bob, dave).

// Rule 1: Child is reverse of parent
rule child(Y, X) :- parent(X, Y).

// Rule 2: Siblings share a parent
rule sibling(X, Y) :- parent(Z, X), parent(Z, Y).

// Rule 3: Grandparent
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

query child(bob, X)?
query sibling(bob, X)?
query grandparent(alice, X)?
```

All rules are applied during forward chaining.

## Step 5: Query with Rules

You can query with rules automatically applied:

```python
from vsar.language.ast import Directive, Fact, Query, Rule, Atom
from vsar.semantics.engine import VSAREngine

# Configure engine
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 1024, "seed": 42}),
    Directive(name="beam", params={"width": 50}),
]
engine = VSAREngine(directives)

# Insert base facts
engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

# Define rules
rules = [
    Rule(
        head=Atom(predicate="ancestor", args=["X", "Y"]),
        body=[Atom(predicate="parent", args=["X", "Y"])],
    ),
    Rule(
        head=Atom(predicate="ancestor", args=["X", "Z"]),
        body=[
            Atom(predicate="parent", args=["X", "Y"]),
            Atom(predicate="ancestor", args=["Y", "Z"]),
        ],
    ),
]

# Query with automatic rule application
query = Query(predicate="ancestor", args=["alice", None])
result = engine.query(query, rules=rules, k=10)

for entity, score in result.results:
    print(f"{entity}: {score:.4f}")
```

## Step 6: Configuration

### Beam Width

Controls candidate bindings in joins:

```prolog
@beam(width=50);    // Default: keep top-50 candidates
@beam(width=100);   // More candidates, slower but more complete
@beam(width=20);    // Fewer candidates, faster
```

### Novelty Threshold

Prevents duplicate derived facts:

```prolog
@novelty(threshold=0.95);   // Default: 95% similarity = duplicate
@novelty(threshold=0.99);   // Stricter (more facts stored)
@novelty(threshold=0.90);   // Looser (fewer facts stored)
```

## Common Rule Patterns

### Pattern 1: Reverse Relationship

```prolog
rule child(Y, X) :- parent(X, Y).
rule employee_of(Y, X) :- employs(X, Y).
```

### Pattern 2: Derived Relationship

```prolog
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
rule uncle(X, Z) :- sibling(X, Y), parent(Y, Z).
```

### Pattern 3: Transitive Closure

```prolog
rule reachable(X, Y) :- edge(X, Y).
rule reachable(X, Z) :- edge(X, Y), reachable(Y, Z).
```

### Pattern 4: Symmetric Relationship

```prolog
rule friend(Y, X) :- friend(X, Y).
rule coauthor(Y, X) :- coauthor(X, Y).
```

## Exercises

1. **Grandchildren:** Write a rule to find grandchildren (reverse of grandparent)
2. **Cousins:** Write rules to find cousins (children of siblings)
3. **Degrees of Separation:** Model social network with friend-of-friend rules

## Debugging Rules

### Check What Was Derived

```python
# Get KB statistics
stats = engine.stats()
print(stats)

# Get all facts for a predicate
facts = engine.kb.get_facts("ancestor")
for fact in facts:
    print(fact)
```

### Enable Tracing

```python
# After query
trace = engine.trace.get_dag()
chaining_events = [e for e in trace if e.type == "chaining"]
for event in chaining_events:
    print(f"Iterations: {event.payload['iterations']}")
    print(f"Derived: {event.payload['total_derived']}")
    print(f"Per iteration: {event.payload['derived_per_iteration']}")
```

### CLI Trace

```bash
vsar run program.vsar --trace
```

## Next Steps

- **[Transitive Closure](transitive-closure.md)** - Deep dive into recursive rules
- **[Knowledge Graphs](knowledge-graphs.md)** - Complex multi-relation reasoning
- **[Advanced Patterns](advanced-patterns.md)** - Expert techniques
- **[Rules & Chaining Guide](../guides/rules-and-chaining.md)** - Complete reference

## Troubleshooting

**Q: My recursive rule doesn't stop!**
A: Set `max_iterations` in forward chaining. Default is 100.

**Q: I'm getting duplicate derived facts**
A: Lower the novelty threshold (e.g., `@novelty 0.90`).

**Q: Results are missing**
A: Increase beam width (e.g., `@beam 100`).
