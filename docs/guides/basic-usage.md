# Basic Usage Guide

This guide covers the fundamentals of using VSAR: defining facts, writing queries, and understanding results.

## Table of Contents

1. [Facts](#facts)
2. [Queries](#queries)
3. [Directives](#directives)
4. [Understanding Results](#understanding-results)
5. [Common Patterns](#common-patterns)
6. [Troubleshooting](#troubleshooting)

---

## Facts

Facts are the building blocks of knowledge in VSAR. They represent ground truth statements about your domain.

### Binary Facts

Most facts are binary relationships between two entities:

```prolog
@model FHRR(dim=8192, seed=42);

fact parent(alice, bob).       // alice is parent of bob
fact parent(bob, carol).        // bob is parent of carol
fact sibling(carol, dave).      // carol and dave are siblings
fact lives_in(alice, boston).   // alice lives in boston
fact works_at(bob, mit).        // bob works at mit
```

### Unary Facts

Unary facts describe properties of single entities:

```prolog
fact person(alice).
fact city(boston).
fact company(mit).
fact active(alice).
```

### Ternary and N-ary Facts

Facts can have more than two arguments:

```prolog
fact transfer(alice, bob, money).           // Ternary
fact meeting(alice, bob, monday, 2pm).      // Quaternary
fact edge(node1, node2, weight, directed).  // Quaternary
```

### Naming Conventions

**Good practices:**

```prolog
fact parent_of(alice, bob).      // Clear relationship
fact employed_by(alice, mit).    // Clear direction
fact born_in(alice, 1990).       // Clear meaning
```

**Avoid:**

```prolog
fact rel(alice, bob).         // Too vague
fact a(b, c).                 // Not descriptive
fact AliceBob(parent, rel).   // Inconsistent case
```

---

## Queries

Queries let you ask questions about your knowledge base. In Phase 1, queries must have exactly one variable.

### Basic Query Patterns

**Pattern 1: Find related entities (forward)**

```prolog
fact parent(alice, bob).
fact parent(alice, carol).

query parent(alice, X)?
// Returns: bob, carol
```

**Pattern 2: Find related entities (backward)**

```prolog
fact parent(alice, bob).
fact parent(charlie, bob).

query parent(X, bob)?
// Returns: alice, charlie
```

**Pattern 3: Property queries**

```prolog
fact lives_in(alice, boston).
fact lives_in(bob, boston).
fact lives_in(carol, nyc).

query lives_in(X, boston)?
// Returns: alice, bob
```

### Ternary Query Patterns

With three arguments, you can query any position:

```prolog
fact transfer(alice, bob, money).
fact transfer(alice, carol, book).
fact transfer(charlie, bob, money).

query transfer(alice, X, money)?    // Who did alice send money to? → bob
query transfer(X, bob, money)?      // Who sent money to bob? → alice, charlie
query transfer(alice, bob, X)?      // What did alice send bob? → money
```

### Variable Naming

Variables must start with uppercase:

```prolog
query parent(alice, X)?       // ✓ Valid
query parent(alice, Person)?  // ✓ Valid
query parent(alice, WHO)?     // ✓ Valid

query parent(alice, x)?       // ✗ Invalid (lowercase)
query parent(alice, _)?       // ✗ Invalid (underscore)
```

---

## Directives

Directives configure how VSAR processes your knowledge base.

### @model Directive

The `@model` directive is required and configures the VSA backend:

```prolog
@model FHRR(dim=8192, seed=42);
```

**Parameters:**

- `dim` - Vector dimensionality (higher = more accurate, slower)
  - Small KBs (< 1000 facts): 1024-4096
  - Medium KBs (1000-10000): 4096-8192
  - Large KBs (> 10000): 8192-16384

- `seed` - Random seed for reproducibility
  - Use same seed for deterministic results
  - Change seed if results seem biased

**Examples:**

```prolog
// Small, fast configuration
@model FHRR(dim=1024, seed=42);

// Balanced (recommended)
@model FHRR(dim=8192, seed=42);

// High accuracy
@model FHRR(dim=16384, seed=42);

// Alternative backend (MAP)
@model MAP(dim=8192, seed=100);
```

### @threshold Directive

Controls similarity threshold for retrieval:

```prolog
@threshold(value=0.22);  // Default
```

**Tuning guidelines:**

- **Lower threshold (0.15-0.20)**: More results, lower precision
  - Use when you want comprehensive recall
  - Example: Exploratory queries

- **Medium threshold (0.22-0.28)**: Balanced
  - Recommended starting point
  - Good for most use cases

- **Higher threshold (0.30-0.40)**: Fewer results, higher precision
  - Use when you want only high-confidence matches
  - Example: Critical decisions

**Example:**

```prolog
@model FHRR(dim=8192, seed=42);
@threshold(value=0.20);  // More permissive

fact parent(alice, bob).
fact parent(alice, carol).
fact parent(bob, dave).

query parent(alice, X)?
// May return more candidates with lower scores
```

---

## Understanding Results

Query results include entities and similarity scores.

### Score Interpretation

Scores range from 0.0 to 1.0:

- **0.9-1.0**: Very high confidence (near-perfect match)
- **0.7-0.9**: High confidence (strong match)
- **0.5-0.7**: Medium confidence (likely match)
- **0.3-0.5**: Low confidence (possible match)
- **< 0.3**: Very low confidence (unlikely match)

**Example output:**

```
┌─────────────────────────┐
│ Query: parent(alice, X) │
├────────┬────────────────┤
│ Entity │ Score          │
├────────┼────────────────┤
│ bob    │ 0.9234         │  ← Very high confidence
│ carol  │ 0.9156         │  ← Very high confidence
│ dave   │ 0.4521         │  ← Low confidence (likely noise)
└────────┴──────────────────┘
```

### Filtering Results

Use scores to filter results programmatically:

```python
from vsar.semantics.engine import VSAREngine
from vsar.language.ast import Query, Directive

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert facts...

query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=10)

# Filter high-confidence results
high_confidence = [(entity, score) for entity, score in result.results if score > 0.8]
print(f"High-confidence matches: {high_confidence}")
```

---

## Common Patterns

### Pattern 1: Family Trees

```prolog
@model FHRR(dim=8192, seed=42);
@threshold(value=0.22);

// Parent relationships
fact parent(grandma, mom).
fact parent(grandma, uncle).
fact parent(mom, alice).
fact parent(mom, bob).
fact parent(uncle, charlie).

// Queries
query parent(mom, X)?           // mom's children → alice, bob
query parent(X, alice)?         // alice's parents → mom
query parent(grandma, X)?       // grandma's children → mom, uncle
```

### Pattern 2: Social Networks

```prolog
@model FHRR(dim=8192, seed=42);

// Friendship (symmetric, but stored both ways)
fact friend(alice, bob).
fact friend(bob, alice).
fact friend(bob, carol).
fact friend(carol, bob).

// Following (asymmetric)
fact follows(alice, bob).
fact follows(alice, carol).
fact follows(bob, carol).

query friend(alice, X)?     // alice's friends
query follows(alice, X)?    // who alice follows
query follows(X, carol)?    // carol's followers
```

### Pattern 3: Knowledge Graphs

```prolog
@model FHRR(dim=8192, seed=42);

// Entities and properties
fact person(alice).
fact person(bob).
fact city(boston).
fact company(mit).

// Relationships
fact lives_in(alice, boston).
fact works_at(alice, mit).
fact works_at(bob, mit).
fact located_in(mit, boston).

// Queries
query works_at(X, mit)?           // MIT employees
query lives_in(X, boston)?        // Boston residents
query located_in(X, boston)?      // What's in Boston
```

### Pattern 4: Temporal Data

```prolog
@model FHRR(dim=8192, seed=42);

fact event(meeting1, monday, 9am).
fact event(meeting2, monday, 2pm).
fact event(meeting3, tuesday, 10am).

fact attended(alice, meeting1).
fact attended(bob, meeting1).
fact attended(alice, meeting2).

query event(X, monday, 9am)?      // Events on Monday at 9am
query attended(alice, X)?         // Meetings alice attended
```

---

## Troubleshooting

### Problem: No Results

**Symptoms:**

```
Query: parent(alice, X)
No results found.
```

**Possible causes:**

1. **No matching facts**: Verify facts exist

```prolog
fact parent(alice, bob).  // Make sure this exists
query parent(alice, X)?
```

2. **Threshold too high**: Lower the threshold

```prolog
@threshold(value=0.15);  // Try lower threshold
```

3. **Wrong predicate name**: Check spelling

```prolog
fact parent(alice, bob).
query parents(alice, X)?  // ✗ Wrong: "parents" vs "parent"
```

### Problem: Low Scores

**Symptoms:**

```
Entity │ Score
bob    │ 0.4521  ← Expected higher
```

**Solutions:**

1. **Increase dimensionality**: Higher dimensions = better accuracy

```prolog
@model FHRR(dim=16384, seed=42);  // Increase from 8192
```

2. **Check for collisions**: Use different seed

```prolog
@model FHRR(dim=8192, seed=100);  // Try different seed
```

3. **Add more context**: More facts improve accuracy

### Problem: Too Many Results

**Symptoms:**

```
Query returned 100 results, most with low scores
```

**Solutions:**

1. **Raise threshold**:

```prolog
@threshold(value=0.30);  // Higher threshold
```

2. **Limit results**:

```bash
vsar run program.vsar --k 5  // Top 5 only
```

3. **Filter programmatically**:

```python
results = [r for r in result.results if r[1] > 0.7]
```

### Problem: Inconsistent Results

**Symptoms:**

```
Same query returns different results on different runs
```

**Solutions:**

1. **Set explicit seed**:

```prolog
@model FHRR(dim=8192, seed=42);  // Fixed seed
```

2. **Use same configuration**: Keep dim and seed consistent

---

## Next Steps

- **[File Formats Guide](file-formats.md)** - Learn CSV, JSONL, VSAR formats
- **[KB Management Guide](kb-management.md)** - Save, load, export KBs
- **[Python API Guide](python-api.md)** - Programmatic usage
- **[Language Reference](../language-reference.md)** - Complete syntax guide

---

## Example: Complete Program

```prolog
/*
  Company Directory

  This KB tracks employees, departments, and locations.
*/

@model FHRR(dim=8192, seed=42);
@threshold(value=0.22);

// Employees
fact person(alice).
fact person(bob).
fact person(carol).

// Departments
fact department(engineering).
fact department(sales).
fact department(hr).

// Employment
fact works_in(alice, engineering).
fact works_in(bob, engineering).
fact works_in(carol, sales).

// Management
fact manages(alice, bob).
fact manages(carol, engineering).

// Locations
fact office_in(engineering, boston).
fact office_in(sales, nyc).

// Queries
query works_in(X, engineering)?   // Who's in engineering?
query manages(alice, X)?          // Who does alice manage?
query office_in(X, boston)?       // What offices are in boston?
query works_in(bob, X)?           // What department is bob in?
```

Run it:

```bash
vsar run company.vsar
```
