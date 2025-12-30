# VSARL Language Reference

VSARL (VSA Reasoning Language) is a declarative language for expressing facts, queries, and rules over knowledge bases using Vector Symbolic Architectures.

## Overview

VSARL programs consist of:

1. **Directives** - Configuration (@model, @threshold, etc.)
2. **Facts** - Ground atoms representing knowledge
3. **Queries** - Questions with variables
4. **Rules** - Horn clauses (Phase 2)

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

Sets beam width for forward chaining (not yet implemented in Phase 1).

**Syntax:**

```prolog
@beam(width=INTEGER);
```

**Example:**

```prolog
@beam(width=50);
```

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

**Phase 1 Limitation:**

- **One variable only** - Multi-variable queries not yet supported
- **Single atom** - Conjunctive queries (multiple atoms) coming in Phase 2

**Invalid in Phase 1:**

```prolog
query parent(X, Y)?                 // Error: 2 variables
query parent(X, bob), parent(bob, Y)?  // Error: conjunction
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

## Rules (Phase 2 - Coming Soon)

Rules define derived relationships using Horn clauses.

**Syntax (Phase 2):**

```prolog
rule PREDICATE(ARGS) :- BODY.
```

**Examples (not yet supported):**

```prolog
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
rule sibling(X, Y) :- parent(P, X), parent(P, Y), X != Y.
rule ancestor(X, Y) :- parent(X, Y).
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
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
statement   ::= directive | fact | query

directive   ::= "@" IDENTIFIER (IDENTIFIER "(" params? ")" | "(" params? ")") ";"
fact        ::= "fact" atom "."
query       ::= "query" atom "?"

atom        ::= predicate "(" args? ")"
predicate   ::= LOWER_NAME
args        ::= arg ("," arg)*
arg         ::= constant | variable

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
