# VSARL Syntax Cheat Sheet

Quick reference for writing correct VSAR programs.

## Basic Syntax Rules

### Identifiers

- **Constants**: lowercase (`alice`, `bob`, `boston`)
- **Variables**: Uppercase (`X`, `Y`, `Person`)
- **Predicates**: lowercase (`parent`, `lives_in`)

### Statements

- **Facts**: End with `.`
- **Queries**: End with `?`
- **Directives**: End with `;`
- **Rules**: End with `.`

---

## Directives

### Model Configuration

```prolog
@model FHRR(dim=1024, seed=42);
@model MAP(dim=512, seed=100);
```

**Required format**: `@model TYPE(param=value, ...);`

### Similarity Threshold

```prolog
@threshold(value=0.22);
```

### Beam Width (for rules)

```prolog
@beam(width=50);
```

**Common mistake**: `@beam 50;` ❌
**Correct**: `@beam(width=50);` ✅

### Novelty Threshold (for rules)

```prolog
@novelty(threshold=0.95);
```

**Common mistake**: `@novelty 0.95;` ❌
**Correct**: `@novelty(threshold=0.95);` ✅

---

## Facts

### Positive Facts

```prolog
fact parent(alice, bob).
fact lives_in(alice, boston).
fact transfer(alice, bob, money).
fact person(alice).
```

### Negative Facts (Phase 3)

```prolog
fact ~enemy(alice, bob).      // Alice is NOT an enemy of Bob
fact ~criminal(alice).         // Alice is NOT a criminal
fact ~member(dave, club).      // Dave is NOT a member
```

**Rules**:
- All arguments must be constants (lowercase)
- End with `.`
- No variables allowed
- Use `~` prefix for negative facts

---

## Queries

```prolog
query parent(alice, X)?
query parent(X, bob)?
query lives_in(X, boston)?
```

**Rules**:
- Exactly one variable (uppercase)
- Variable can be in any position
- End with `?`

**Common mistakes**:
- `query parent(?)?` ❌ (use `X` not `?`)
- `query parent(X, Y)?` ❌ (only one variable allowed)

**Correct**:
- `query parent(X)?` ✅
- `query parent(alice, X)?` ✅

---

## Rules (Phase 2)

### Single-body Rule

```prolog
rule human(X) :- person(X).
```

### Multi-body Rule (Join)

```prolog
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

### Recursive Rule

```prolog
rule ancestor(X, Y) :- parent(X, Y).
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

### Multiple Rules

```prolog
rule connected(X, Y) :- knows(X, Y).
rule connected(X, Y) :- works_with(X, Y).
rule connected(X, Z) :- connected(X, Y), connected(Y, Z).
```

**Rules**:
- All variables in HEAD must appear in BODY
- Use `,` for conjunction (AND)
- End with `.`

---

## Negation-as-Failure (Phase 3)

### NAF in Rules

```prolog
rule safe(X) :- person(X), not enemy(X, Y).
```

### Multiple NAF Literals

```prolog
rule trustworthy(X) :-
    person(X),
    not enemy(X, Y),
    not criminal(X).
```

### NAF vs Classical Negation

```prolog
// Classical negation (explicit fact)
fact ~enemy(alice, bob).

// Negation-as-failure (closed-world in rules)
rule safe(X) :- person(X), not enemy(X, Y).
```

**Rules**:
- Use `not` keyword before atom
- NAF only in rule bodies
- Succeeds if atom cannot be proven
- Variables can be wildcards (unbound)

**Common mistakes**:
- `rule safe(X) :- person(X), ~enemy(X, Y).` ❌ (use `not` not `~`)
- `fact not enemy(alice, bob).` ❌ (use `~` for negative facts)

**Correct**:
- `rule safe(X) :- person(X), not enemy(X, Y).` ✅
- `fact ~enemy(alice, bob).` ✅

---

## Complete Example

### Without Negation

```prolog
// Configuration
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

// Queries
query ancestor(alice, X)?
query parent(X, dave)?
```

### With Negation (Phase 3)

```prolog
// Configuration
@model FHRR(dim=1024, seed=42);

// Positive facts
fact person(alice).
fact person(bob).
fact person(carol).
fact enemy(bob, dave).

// Negative facts (explicit)
fact ~criminal(alice).
fact ~criminal(carol).

// Rules with NAF
rule safe(X) :-
    person(X),
    not enemy(X, Y).

rule trustworthy(X) :-
    safe(X),
    not criminal(X).

// Queries
query safe(X)?           // Returns: alice, carol
query trustworthy(X)?    // Returns: alice, carol
```

---

## Common Errors

### ❌ Wrong Directive Syntax

```prolog
@beam 50;
@novelty 0.95;
```

### ✅ Correct Directive Syntax

```prolog
@beam(width=50);
@novelty(threshold=0.95);
```

### ❌ Wrong Query Variable

```prolog
query parent(?)?
query parent(alice, ?)?
```

### ✅ Correct Query Variable

```prolog
query parent(X)?
query parent(alice, X)?
```

### ❌ Multiple Variables

```prolog
query parent(X, Y)?  // Not supported yet
```

### ✅ Single Variable

```prolog
query parent(X, bob)?
query parent(alice, X)?
```

---

## Comments

```prolog
// Single-line comment

/* Multi-line
   comment */

fact parent(alice, bob).  // Inline comment
```

---

## See Also

- **[Language Reference](language-reference.md)** - Complete syntax guide
- **[Getting Started](getting-started.md)** - First steps
- **[Rules & Chaining Guide](guides/rules-and-chaining.md)** - Rule details
- **[Examples](examples.md)** - Working programs
