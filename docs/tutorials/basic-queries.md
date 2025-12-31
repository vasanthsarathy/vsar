# Tutorial: Basic Facts & Queries

This tutorial introduces the fundamentals of VSAR: inserting facts and executing queries.

## What You'll Learn

- How to define facts
- How to write queries
- Understanding similarity scores
- Working with different arities

## Prerequisites

- VSAR installed (`pip install vsar`)
- Basic understanding of predicates and arguments

## Step 1: Your First Fact

Facts are the building blocks of VSAR. They represent ground truth in your knowledge base.

Create a file `tutorial1.vsar`:

```prolog
@model FHRR(dim=1024, seed=42);

fact person(alice).
```

Run it:

```bash
vsar run tutorial1.vsar
```

Output:
```
Inserted 1 fact
```

**Explanation:**
- `@model` configures the VSA backend (FHRR with 1024 dimensions)
- `fact person(alice).` declares that alice is a person
- Facts end with a period (`.`)

## Step 2: Binary Facts

Most facts have two arguments (binary predicates):

```prolog
@model FHRR(dim=1024, seed=42);

fact parent(alice, bob).
fact parent(bob, carol).
fact parent(carol, dave).
```

**Structure:** `predicate(arg1, arg2)`

- `parent(alice, bob)` means "alice is parent of bob"
- Order matters: `parent(bob, alice)` is different!

## Step 3: Your First Query

Queries retrieve facts from the KB. Variables (uppercase) are placeholders.

```prolog
@model FHRR(dim=1024, seed=42);

fact parent(alice, bob).
fact parent(alice, carol).

query parent(alice, X)?
```

Run it:

```bash
vsar run tutorial1.vsar
```

Output:
```
Inserted 2 facts

┌─────────────────────────┐
│ Query: parent(alice, X) │
├────────┬────────────────┤
│ Entity │ Score          │
├────────┼────────────────┤
│ bob    │ 0.9234         │
│ carol  │ 0.9156         │
└────────┴────────────────┘
```

**Explanation:**
- `X` is a variable (will be bound to results)
- `alice` is bound (fixed value)
- Query returns all values for `X` where `parent(alice, X)` is true

## Step 4: Different Query Positions

You can query any position:

```prolog
@model FHRR(dim=1024, seed=42);

fact parent(alice, bob).
fact parent(carol, bob).

// Query 1: Who are alice's children?
query parent(alice, X)?

// Query 2: Who are bob's parents?
query parent(X, bob)?
```

Output:
```
Query: parent(alice, X)
  bob: 0.9234

Query: parent(X, bob)
  alice: 0.8876
  carol: 0.8654
```

## Step 5: Understanding Scores

VSAR uses **similarity-based retrieval**, not exact matching.

**What scores mean:**
- `1.0` = Perfect match (rare due to noise)
- `0.9+` = Very strong match (typical)
- `0.8+` = Good match
- `0.7+` = Weak match
- `< 0.7` = Probably wrong

**Why approximate?**
- Graceful degradation under noise
- Fuzzy matching (typos, similar names)
- Vector-based operations

## Step 6: Ternary Facts

Facts can have 3+ arguments:

```prolog
@model FHRR(dim=1024, seed=42);

fact transfer(alice, bob, money).
fact transfer(bob, carol, book).
fact transfer(carol, dave, key).

query transfer(alice, X, money)?
query transfer(X, carol, Y)?
```

Output:
```
Query: transfer(alice, X, money)
  bob: 0.9123

Query: transfer(X, carol, Y)
  bob, book: 0.8945
```

## Step 7: Multiple Queries

You can include multiple queries in one program:

```prolog
@model FHRR(dim=1024, seed=42);

fact lives_in(alice, boston).
fact lives_in(bob, cambridge).
fact lives_in(carol, boston).
fact works_at(alice, mit).
fact works_at(bob, harvard).
fact works_at(carol, mit).

query lives_in(X, boston)?
query works_at(X, mit)?
query lives_in(alice, X)?
```

Each query executes independently and returns results.

## Step 8: Limiting Results

Control how many results to return:

```bash
# Return top 5 results per query
vsar run tutorial1.vsar --k 5
```

Or in the program:

```prolog
@model FHRR(dim=1024, seed=42);
@beam(width=50);  // Affects rule processing (Phase 2)

// Queries return top-k results (default: 10)
```

## Common Patterns

### Pattern 1: Existence Check

```prolog
fact person(alice).
query person(alice)?  // Returns high score if exists
```

### Pattern 2: Finding Relationships

```prolog
fact friend(alice, bob).
fact friend(bob, carol).

query friend(alice, X)?  // Who are alice's friends?
```

### Pattern 3: Reverse Lookup

```prolog
fact employee(alice, acme_corp).
fact employee(bob, acme_corp).

query employee(X, acme_corp)?  // Who works at acme_corp?
```

## Python API

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

# Configure engine
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 1024, "seed": 42})
]
engine = VSAREngine(directives)

# Insert facts
engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

# Execute query
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=10)

# Process results
for entity, score in result.results:
    print(f"{entity}: {score:.4f}")
```

## Exercises

1. **Family Database:** Create facts for your family tree (parents, siblings)
2. **Social Network:** Model friend relationships
3. **Organization:** Model company structure (employees, departments)

## Next Steps

- **[Rules & Reasoning](rules-and-reasoning.md)** - Learn to derive new facts
- **[Transitive Closure](transitive-closure.md)** - Multi-hop reasoning
- **[Language Reference](../language-reference.md)** - Complete syntax

## Troubleshooting

**Q: Why are my scores not 1.0?**
A: VSAR uses approximate VSA-based matching. Scores of 0.9+ are excellent.

**Q: Can I query with no bound arguments?**
A: Not yet. `query parent(?, ?)?` is planned for Phase 3.

**Q: How many facts can VSAR handle?**
A: Millions. Performance scales linearly. See [Performance Guide](../guides/performance.md).
