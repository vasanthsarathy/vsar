# VSARL Language Reference

VSARL (VSA Reasoning Language) is a declarative language for expressing facts, queries, and rules over knowledge bases using Vector Symbolic Architectures.

## Overview

VSARL programs consist of:

1. **Directives** - Configuration (@model, @threshold, etc.)
2. **Facts** - Ground atoms representing knowledge (positive and negative)
3. **Queries** - Questions with variables
4. **Rules** - Horn clauses with optional negation-as-failure (Phase 2 + Phase 3)

## Syntax Conventions

- **Constants**: Lowercase identifiers (`alice`, `bob`, `boston`)
- **Variables**: Uppercase identifiers (`X`, `Y`, `Person`)
- **Predicates**: Lowercase identifiers (`parent`, `lives_in`)
- **Comments**: `//` for single-line, `/* */` for multi-line
- **Statements**: Terminated with `.` (facts) or `?` (queries) or `;` (directives)

---

## Directives

Directives configure the reasoning engine. They start with `@` and end with `;`.

### @model

Configures the VSA backend and parameters.

**Syntax:**

```prolog
@model TYPE(dim=DIMENSIONS, seed=SEED);
```

**Parameters:**

- `TYPE` - Backend type: `FHRR` (default) or `MAP`
- `dim` - Vector dimensionality (default: 8192)
- `seed` - Random seed for determinism (default: 42)

**Examples:**

```prolog
@model FHRR(dim=8192, seed=42);
@model MAP(dim=4096, seed=100);
@model FHRR(dim=16384, seed=1);
```

**Valid Dimensions:**

- Typical: 512, 1024, 2048, 4096, 8192, 16384
- Higher dimensions = better accuracy, slower queries
- Recommended: 8192 for most use cases

---

### @threshold

Sets the similarity threshold for retrieval.

**Syntax:**

```prolog
@threshold(value=FLOAT);
```

**Parameters:**

- `value` - Similarity threshold (0.0 to 1.0, default: 0.22)

**Examples:**

```prolog
@threshold(value=0.22);  // Default
@threshold(value=0.5);   // Stricter matching
@threshold(value=0.1);   // Looser matching
```

**Guidelines:**

- Lower threshold = more results (higher recall)
- Higher threshold = fewer results (higher precision)
- Typical range: 0.15 to 0.35
- Experiment with your dataset to find optimal value

---

### @beam (Phase 2)

Sets beam width for forward chaining joins.

**Syntax:**

```prolog
@beam(width=INTEGER);
```

**Parameters:**

- `width` - Number of candidate bindings to keep during joins (default: 50)

**Examples:**

```prolog
@beam(width=50);   // Default
@beam(width=100);  // More candidates, slower but more complete
@beam(width=20);   // Fewer candidates, faster
```

**Guidelines:**

- Small KB (<1K facts): 20-50
- Medium KB (1K-10K): 50-100
- Large KB (>10K): 100-200

---

### @novelty (Phase 2)

Sets novelty detection threshold to prevent duplicate derived facts.

**Syntax:**

```prolog
@novelty(threshold=FLOAT);
```

**Parameters:**

- `threshold` - Similarity threshold for duplicate detection (0.0 to 1.0, default: 0.95)

**Examples:**

```prolog
@novelty(threshold=0.95);  // Default (balanced)
@novelty(threshold=0.99);  // Stricter (more facts, slower)
@novelty(threshold=0.90);  // Looser (fewer facts, faster)
```

**Guidelines:**

- Higher threshold = stricter duplicate detection
- Lower threshold = looser duplicate detection
- Typical range: 0.90 to 0.99

---

## Facts

Facts are ground atoms (all arguments are constants).

**Syntax:**

```prolog
fact PREDICATE(ARG1, ARG2, ...).
```

**Examples:**

**Binary facts (most common):**

```prolog
fact parent(alice, bob).
fact parent(bob, carol).
fact lives_in(alice, boston).
fact works_at(bob, harvard).
```

**Unary facts:**

```prolog
fact person(alice).
fact city(boston).
fact company(mit).
```

**Ternary facts:**

```prolog
fact transfer(alice, bob, money).
fact meeting(alice, bob, monday).
fact edge(node1, node2, weight5).
```

**N-ary facts:**

```prolog
fact relation(a, b, c, d, e).
```

**Constraints:**

- All arguments must be constants (lowercase)
- No variables allowed in facts
- No spaces in predicate names (use `_` instead)
- Arguments are comma-separated
- Statement ends with `.`

---

## Negative Facts (Phase 3)

VSARL supports **classical negation** for explicitly representing negative knowledge.

**Syntax:**

```prolog
fact ~PREDICATE(ARG1, ARG2, ...).
```

The `~` prefix marks a fact as negative (strong negation).

**Examples:**

```prolog
// Positive facts
fact friend(alice, bob).
fact friend(bob, carol).

// Negative facts (explicit negative knowledge)
fact ~enemy(alice, bob).     // Alice is NOT an enemy of Bob
fact ~criminal(alice).        // Alice is NOT a criminal
fact ~member(dave, club).     // Dave is NOT a member of the club
```

**Use Cases:**

1. **Closed-World Knowledge**: Explicitly state what is NOT true
2. **Contradiction Detection**: System can detect inconsistencies
3. **Paraconsistent Reasoning**: KB can contain both `p(a)` and `~p(a)` without failing

**Querying Negative Facts:**

```prolog
// Query negative facts using ~ prefix
query ~enemy(alice, X)?       // Who is alice NOT enemies with?
query ~criminal(X)?           // Who is NOT a criminal?
```

**Consistency Checking:**

VSAR operates in **paraconsistent mode** - it allows contradictions:

```prolog
fact friend(alice, bob).
fact ~friend(alice, bob).     // Contradiction allowed!

// System warns but continues execution
// Consistency check will report the contradiction
```

To check for contradictions:

```python
from vsar.reasoning import ConsistencyChecker

checker = ConsistencyChecker(engine.kb)
report = checker.check()

if not report.is_consistent:
    print(report.summary())
```

**Important Notes:**

- `~p(a)` (negative fact) is different from `not p(a)` (negation-as-failure)
- Negative facts are explicitly stored in the KB
- Use negative facts when you have **explicit negative knowledge**
- Use NAF (see Rules section) when you want **closed-world assumption**

---

## Queries

Queries are atoms with exactly one variable (Phase 1 limitation).

**Syntax:**

```prolog
query PREDICATE(ARG1, ARG2, ..., X, ...)?
```

where exactly one argument is a variable (uppercase identifier).

**Examples:**

**Find children of alice:**

```prolog
query parent(alice, X)?
```

**Find parents of bob:**

```prolog
query parent(X, bob)?
```

**Who lives in boston:**

```prolog
query lives_in(X, boston)?
```

**Where does alice live:**

```prolog
query lives_in(alice, X)?
```

**Ternary queries:**

```prolog
query transfer(alice, X, money)?    // Alice transferred money to whom?
query transfer(X, bob, money)?      // Who transferred money to bob?
query transfer(alice, bob, X)?      // What did alice transfer to bob?
```

**Current Limitations:**

- **One variable only** - Multi-variable queries not yet supported (planned for Phase 3)
- **Single atom** - Conjunctive queries (multiple atoms) not yet supported

**Invalid:**

```prolog
query parent(X, Y)?                    // Error: 2 variables (Phase 3)
query parent(X, bob), parent(bob, Y)?  // Error: conjunction (use rules instead)
```

**Note:** For multi-hop reasoning, use rules instead of conjunctive queries:

```prolog
// Instead of: query parent(X, bob), parent(bob, Y)?
// Use rules:
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
query grandparent(alice, X)?
```

---

## Comments

VSARL supports single-line and multi-line comments.

**Single-line comments:**

```prolog
// This is a single-line comment
fact parent(alice, bob).  // Inline comment
```

**Multi-line comments:**

```prolog
/*
  This is a multi-line comment.
  It can span multiple lines.
*/
fact parent(alice, bob).

/* Inline block comment */ fact sibling(bob, carol).
```

---

## Complete Program Example

```prolog
/*
  Family Tree Knowledge Base
  Author: VSAR User
  Date: 2025-01-15
*/

// Configure VSA backend
@model FHRR(dim=8192, seed=42);
@threshold(value=0.22);

// Facts: Parent relationships
fact parent(alice, bob).
fact parent(alice, carol).
fact parent(bob, dave).
fact parent(bob, eve).
fact parent(carol, frank).

// Facts: Location information
fact lives_in(alice, boston).
fact lives_in(bob, cambridge).
fact lives_in(carol, boston).
fact lives_in(dave, nyc).

// Facts: Employment
fact works_at(alice, mit).
fact works_at(bob, harvard).
fact works_at(carol, mit).

// Queries: Who are alice's children?
query parent(alice, X)?

// Queries: Where do people live?
query lives_in(X, boston)?

// Queries: Who works at MIT?
query works_at(X, mit)?
```

---

## Rules (Phase 2)

Rules define derived relationships using Horn clauses. **Phase 2 is now complete!**

**Syntax:**

```prolog
rule HEAD :- BODY1, BODY2, ..., BODYN.
```

**Components:**

- `HEAD` - Single atom (the derived fact)
- `BODY1, BODY2, ...` - One or more atoms (the conditions)
- Variables in HEAD must appear in BODY
- Comma `,` represents conjunction (AND)

**Examples:**

**Single-body rule:**
```prolog
rule human(X) :- person(X).
```

**Multi-body rule (join):**
```prolog
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

**Recursive rules (transitive closure):**
```prolog
// Base case
rule ancestor(X, Y) :- parent(X, Y).

// Recursive case
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

**Multiple rules with same head:**
```prolog
// Combine different sources
rule connected(X, Y) :- knows(X, Y).
rule connected(X, Y) :- works_with(X, Y).
rule connected(X, Z) :- connected(X, Y), connected(Y, Z).
```

**Configuration directives for rules:**

```prolog
@beam(width=50);           // Beam width for joins (default: 50)
@novelty(threshold=0.95);  // Novelty detection threshold (default: 0.95)
```

**Complete example:**

```prolog
@model FHRR(dim=1024, seed=42);
@beam(width=50);
@novelty(threshold=0.95);

// Base facts
fact parent(alice, bob).
fact parent(bob, carol).
fact parent(carol, dave).

// Rules
rule ancestor(X, Y) :- parent(X, Y).
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

// Query derived facts
query ancestor(alice, X)?
```

See [Rules & Chaining Guide](guides/rules-and-chaining.md) for detailed documentation.

---

## Negation-as-Failure (Phase 3)

VSARL supports **negation-as-failure (NAF)** in rule bodies using the `not` keyword.

**Syntax:**

```prolog
rule HEAD :- BODY1, BODY2, not BODY3, ...
```

NAF implements the **closed-world assumption**: `not p(X)` succeeds if `p(X)` cannot be proven from the KB.

**Examples:**

**Basic NAF:**

```prolog
// Safe person: someone who is a person and NOT an enemy of anyone
rule safe(X) :- person(X), not enemy(X, Y).

fact person(alice).
fact person(bob).
fact enemy(bob, carol).

// Derives: safe(alice)  ✓
// Does NOT derive: safe(bob)  ✗ (bob has an enemy)
```

**Multiple NAF Literals:**

```prolog
// Trustworthy: safe AND not a criminal
rule trustworthy(X) :-
    person(X),
    not enemy(X, Y),
    not criminal(X).

fact person(alice).
fact person(bob).
fact person(carol).
fact enemy(bob, dave).
fact criminal(carol).

// Derives: trustworthy(alice)  ✓
```

**NAF with Bound Variables:**

```prolog
// People who are NOT friends with specific person
rule not_friends_with_bob(X) :-
    person(X),
    not friend(X, bob).

fact person(alice).
fact person(bob).
fact person(carol).
fact friend(alice, bob).

// Derives: not_friends_with_bob(carol)  ✓
```

**Wildcard Variables (Existential Check):**

```prolog
// X has NO enemies (wildcard Y means "any")
not enemy(X, Y)          // True if enemy(X, _) has no matches

// X is NOT a member of ANY club
not member(X, Club)      // True if member(X, _) has no matches
```

**NAF vs Classical Negation:**

| Feature | NAF (`not p(X)`) | Classical Negation (`~p(a)`) |
|---------|------------------|------------------------------|
| **Type** | Closed-world assumption | Explicit negative knowledge |
| **Usage** | In rule bodies only | As facts or queries |
| **Semantics** | Succeeds if unprovable | Explicit negative assertion |
| **Storage** | Not stored (computed) | Stored in KB |

**Example showing difference:**

```prolog
// Classical negation (explicit)
fact ~enemy(alice, bob).     // Alice is explicitly NOT an enemy of Bob

// Negation-as-failure (closed-world)
rule safe(X) :- person(X), not enemy(X, Y).
// "If we can't prove X has enemies, X is safe"
```

**Stratification:**

VSAR automatically checks for **stratified** NAF usage:

```prolog
// ✓ STRATIFIED (safe)
rule safe(X) :- person(X), not enemy(X, Y).
rule trusted(X) :- safe(X), not criminal(X).

// ✗ NON-STRATIFIED (warning)
rule p(X) :- not q(X).
rule q(X) :- not p(X).  // Circular negative dependency!
```

Non-stratified programs trigger a warning:

```
Warning: Non-stratified program detected.
Negation-as-failure may have unpredictable semantics.
✗ Program is NOT stratified - contains negative cycles:
  Cycle 1: p → q → p
```

VSAR will continue execution but results may be unpredictable.

**Best Practices:**

1. **Use NAF for closed-world reasoning**: "Not provable = false"
2. **Use classical negation for explicit knowledge**: "Known to be false"
3. **Avoid circular NAF dependencies**: Keep rules stratified
4. **Test NAF behavior**: NAF semantics can be subtle

**Complete Example:**

```prolog
@model FHRR(dim=1024, seed=42);

// Base facts
fact person(alice).
fact person(bob).
fact person(carol).
fact enemy(bob, dave).
fact criminal(carol).

// Rules with NAF
rule safe(X) :-
    person(X),
    not enemy(X, Y).

rule trustworthy(X) :-
    safe(X),
    not criminal(X).

// Query
query trustworthy(X)?

// Results: trustworthy(alice) ✓
```

---

## Lexical Structure

### Identifiers

**Constants** (lowercase):

- Start with lowercase letter
- Followed by letters, digits, underscores
- Examples: `alice`, `bob`, `lives_in`, `node_1`

**Variables** (uppercase):

- Start with uppercase letter
- Followed by letters, digits, underscores
- Examples: `X`, `Y`, `Person`, `Location`

**Predicates** (lowercase):

- Same rules as constants
- Examples: `parent`, `lives_in`, `transfer`

### Literals

**Numbers:**

```prolog
@model FHRR(dim=8192, seed=42);
@threshold(value=0.22);
```

**Strings:**

```prolog
@model FHRR(dim=8192, seed=42);
```

**Booleans:**

```prolog
@option(strict=true);
```

---

## Parsing Rules

### Whitespace

Whitespace (spaces, tabs, newlines) is ignored except:

- Inside string literals
- To separate tokens

**Valid:**

```prolog
fact parent(alice, bob).
fact parent(alice,bob).
fact parent ( alice , bob ) .
```

### Case Sensitivity

VSARL is case-sensitive:

- `alice` ≠ `Alice`
- `parent` ≠ `Parent`
- `X` ≠ `x`

### Operator Precedence

**Phase 1:**

- No operators yet

**Phase 2:**

- `,` (conjunction) - binds tightest
- `:-` (implication) - binds loosest

---

## Error Messages

### Syntax Errors

**Unexpected token:**

```prolog
fact Parent(alice, bob).  // Error: Parent starts with uppercase
```

Error:

```
Parse error at line 1: Expected lowercase identifier, got 'Parent'
```

**Missing terminator:**

```prolog
fact parent(alice, bob)   // Error: Missing '.'
```

Error:

```
Parse error at line 1: Expected '.', got end of line
```

**Invalid variable count:**

```prolog
query parent(X, Y)?  // Error: Too many variables
```

Error:

```
Query must have exactly 1 variable, got 2
```

### Semantic Errors

**Undefined predicate:**

```prolog
@model FHRR(dim=8192, seed=42);
query parent(alice, X)?  // Error: No 'parent' facts
```

Error:

```
Predicate 'parent' not found in knowledge base
```

---

## Grammar (EBNF)

Complete grammar in Extended Backus-Naur Form:

```ebnf
program     ::= statement*
statement   ::= directive | fact | query | rule

directive   ::= "@" IDENTIFIER (IDENTIFIER "(" params? ")" | "(" params? ")") ";"
fact        ::= "fact" "~"? atom "."
query       ::= "query" "~"? atom "?"
rule        ::= "rule" atom ":-" body "."

atom        ::= predicate "(" args? ")"
predicate   ::= LOWER_NAME
args        ::= arg ("," arg)*
arg         ::= constant | variable

body        ::= body_literal ("," body_literal)*
body_literal::= atom | naf_literal
naf_literal ::= "not" atom

constant    ::= LOWER_NAME
variable    ::= UPPER_NAME

params      ::= param ("," param)*
param       ::= IDENTIFIER "=" value
value       ::= NUMBER | STRING | TRUE | FALSE | IDENTIFIER

LOWER_NAME  ::= [a-z][a-z0-9_]*
UPPER_NAME  ::= [A-Z][A-Z0-9_]*
IDENTIFIER  ::= [a-zA-Z][a-zA-Z0-9_]*
NUMBER      ::= [0-9]+ ("." [0-9]+)?
STRING      ::= '"' [^"]* '"'
TRUE        ::= "true"
FALSE       ::= "false"

comment     ::= "//" [^\n]* | "/*" .* "*/"
```

---

## Best Practices

1. **Use meaningful names**: `lives_in` instead of `rel1`
2. **Consistent naming**: Choose a convention (snake_case recommended)
3. **Add comments**: Document complex facts or queries
4. **Set appropriate seed**: Use same seed for reproducibility
5. **Tune threshold**: Experiment to find optimal value
6. **Group related facts**: Organize facts by predicate
7. **Test incrementally**: Start small, add facts gradually

**Example:**

```prolog
// Good: Clear, well-organized
@model FHRR(dim=8192, seed=42);

// Parent relationships
fact parent(alice, bob).
fact parent(bob, carol).

// Location data
fact lives_in(alice, boston).

// Bad: Unclear, poorly organized
@model FHRR(dim=512, seed=1);
fact r1(a,b).
fact r2(a,c).
fact r1(b,c).
```

---

## See Also

- [Getting Started](getting-started.md) - Quick start guide
- [CLI Reference](cli-reference.md) - Command-line usage
- [User Guides](guides/basic-usage.md) - Detailed tutorials
- [Architecture](architecture.md) - How VSAR works
