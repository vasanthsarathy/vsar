# Negation in VSAR: A Comprehensive Guide

This guide provides a complete introduction to negation in VSAR, covering both classical negation and negation-as-failure (NAF).

## Table of Contents

1. [Introduction](#introduction)
2. [Classical Negation](#classical-negation)
3. [Negation-as-Failure](#negation-as-failure)
4. [Comparison: Classical vs NAF](#comparison-classical-vs-naf)
5. [Stratification](#stratification)
6. [Best Practices](#best-practices)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

## Introduction

VSAR supports two forms of negation:

1. **Classical Negation (`~`)**: Explicit negative facts stored in the knowledge base
2. **Negation-as-Failure (`not`)**: Closed-world assumption in rule bodies

These two forms serve different purposes and can be used together to express complex reasoning patterns.

## Classical Negation

### What is Classical Negation?

Classical negation allows you to explicitly state that something is **NOT** true. These are stored as facts in the knowledge base, just like positive facts.

### Syntax

```prolog
fact ~PREDICATE(ARG1, ARG2, ...).
```

### Examples

```prolog
// Explicit negative facts
fact ~enemy(alice, bob).       // Alice is NOT an enemy of Bob
fact ~criminal(alice).          // Alice is NOT a criminal
fact ~member(dave, club_a).     // Dave is NOT a member of club_a
```

### When to Use

Use classical negation when:
- You have **explicit knowledge** that something is false
- The negation comes from external data sources
- You want to represent exceptions or exclusions
- You need to distinguish "known to be false" from "unknown"

### Paraconsistent Logic

VSAR allows **contradictions** - you can have both `fact p(a)` and `fact ~p(a)` in the knowledge base without system failure. This is called paraconsistent logic.

```prolog
fact criminal(bob).        // Bob is a criminal
fact ~criminal(bob).       // Bob is also marked as NOT a criminal

// Both facts coexist - VSAR does not fail
// Queries will return both positive and negative results
```

**Use case**: Representing conflicting information from multiple sources.

### Querying Negative Facts

You can query for negative facts just like positive facts:

```prolog
// Check if alice is NOT a criminal
query ~criminal(alice)?

// Find all people who are NOT enemies of bob
query ~enemy(X, bob)?
```

### Consistency Checking

While VSAR allows contradictions, you may want to detect them. The consistency checker can identify contradictions:

```python
from vsar.reasoning.consistency import check_consistency

# Check for contradictions
result = check_consistency(engine.kb)

if not result.is_consistent:
    for contradiction in result.contradictions:
        print(f"Contradiction: {contradiction.predicate}{contradiction.args}")
```

## Negation-as-Failure

### What is Negation-as-Failure?

Negation-as-failure (NAF) implements the **closed-world assumption**: if something cannot be proven true, it is assumed false.

### Syntax

```prolog
rule HEAD :- BODY1, BODY2, not BODY3, BODY4.
```

NAF can only appear in **rule bodies**, not in facts or rule heads.

### Examples

```prolog
// A person is safe if they are a person and have no enemies
rule safe(X) :-
    person(X),
    not enemy(X, Y).

// A person is eligible if they are a member and not banned
rule eligible(X) :-
    member(X),
    not banned(X).

// A route is available if it exists and is not blocked
rule available(X, Y) :-
    route(X, Y),
    not blocked(X, Y).
```

### When to Use

Use negation-as-failure when:
- You want to express "absence of information" as false
- Implementing default reasoning ("typically true unless...")
- Expressing integrity constraints
- Filtering results based on lack of certain properties

### How NAF Works

NAF evaluates by checking if a fact **cannot be derived**:

1. Apply current variable bindings to the NAF literal
2. Check if the resulting atom exists in the KB (with exact matching on bound arguments)
3. If found: NAF **fails** (returns False)
4. If not found: NAF **succeeds** (returns True)

### Wildcards in NAF

Unbound variables (uppercase) in NAF literals act as **existential wildcards**:

```prolog
// not enemy(X, Y) succeeds if there exists NO pair (X, Y) matching bound positions
rule safe(alice) :- not enemy(alice, Y).
// Succeeds only if alice has NO enemies at all

// With bound variables
rule friendly(alice, bob) :- not enemy(alice, bob).
// Succeeds only if the specific tuple enemy(alice, bob) does not exist
```

### Exact Matching

NAF uses **exact tuple matching** on bound arguments, not VSA similarity:

```prolog
fact enemy(bob, carol).

// This NAF succeeds (alice != bob, exact match required)
rule safe(alice) :- not enemy(alice, Y).
```

This ensures NAF is precise and predictable.

## Comparison: Classical vs NAF

| Aspect | Classical Negation (`~`) | Negation-as-Failure (`not`) |
|--------|-------------------------|----------------------------|
| **Location** | Facts only | Rule bodies only |
| **Semantics** | Explicit knowledge of falsity | Absence of proof |
| **Storage** | Stored in KB | Computed dynamically |
| **World** | Open-world (both true and false can be unknown) | Closed-world (unknown = false) |
| **Contradictions** | Allowed (paraconsistent) | Not applicable |
| **Query** | Can be queried directly | Cannot be queried directly |
| **Example** | `fact ~enemy(alice, bob).` | `rule safe(X) :- not enemy(X, Y).` |

### Combined Use

Classical negation and NAF can work together:

```prolog
// Explicit negative fact
fact ~criminal(alice).

// NAF rule
rule trustworthy(X) :-
    person(X),
    not enemy(X, Y),      // NAF: no enemies
    not criminal(X).       // NAF: not proven criminal

// Query
query trustworthy(alice)?
// Succeeds if:
// 1. alice is a person
// 2. alice has no enemy facts
// 3. No criminal(alice) fact AND no ~criminal(alice) fact
//    OR ~criminal(alice) exists (which contradicts nothing)
```

## Stratification

### What is Stratification?

Stratification is a property of logic programs that ensures negation has well-defined semantics. A program is **stratified** if it can be partitioned into layers (strata) where:

- Each stratum only depends on lower strata
- Negation only refers to predicates in strictly lower strata

### Why Does It Matter?

Non-stratified programs can have:
- **Ambiguous semantics** - multiple valid interpretations
- **Unstable fixed points** - iteration may not converge
- **Paradoxes** - contradictory derivations

### Stratification Examples

#### ✅ Stratified (Safe)

```prolog
// Stratum 0: Base facts
fact person(alice).
fact person(bob).
fact enemy(bob, carol).

// Stratum 1: safe depends on person and enemy (both stratum 0)
rule safe(X) :-
    person(X),
    not enemy(X, Y).

// Stratum 2: trustworthy depends on safe (stratum 1)
rule trustworthy(X) :-
    safe(X),
    not criminal(X).
```

**Dependency graph**: No cycles through negation.

#### ❌ Non-Stratified (Unsafe)

```prolog
// Cycle through negation!
rule p(X) :- not q(X).
rule q(X) :- not p(X).

// Paradox: If p is true, then q must be false, so p is false!
```

**Dependency graph**: Contains cycle with negative edge.

### Automatic Stratification Analysis

VSAR automatically checks stratification when running programs:

```prolog
// This will generate a warning
rule p(X) :- not q(X).
rule q(X) :- not p(X).
```

**Warning**:
```
Warning: Non-stratified program detected.
Negation-as-failure may have unpredictable semantics.

Negative cycles found:
  - p → q → p

No valid stratification exists.
```

### Handling Non-Stratified Programs

If VSAR detects a non-stratified program:

1. **Warning is issued** - User is notified
2. **Execution continues** - Forward chaining proceeds (may not terminate or may give unexpected results)
3. **Manual review recommended** - Check if the program is correct

**Resolution strategies**:
- Restructure rules to break negative cycles
- Use classical negation for some predicates
- Accept that the program may have multiple models

## Best Practices

### 1. Choose the Right Negation

**Use classical negation (`~`) when**:
- You have explicit negative information from data
- You want to represent exceptions
- You need to query negative facts directly
- You're integrating multiple sources with conflicts

**Use NAF (`not`) when**:
- You want to check for absence
- Implementing default rules ("unless...")
- Filtering candidates in rules
- Expressing integrity constraints

### 2. Keep Programs Stratified

- Avoid negative cycles (predicate depending on itself through negation)
- Layer your rules: base facts → derived facts → negated checks
- Use dependency graphs to visualize rule interactions

### 3. Be Explicit About Unknowns

```prolog
// Clear: explicitly mark alice as not a criminal
fact ~criminal(alice).

// Ambiguous: alice's criminal status is unknown
// (NAF will treat as false)
```

### 4. Document Closed-World Assumptions

When using NAF, document which predicates use closed-world assumption:

```prolog
// Closed-world predicates: enemy, criminal, banned
// If not in KB, assumed false

rule safe(X) :- person(X), not enemy(X, Y).
rule eligible(X) :- member(X), not banned(X).
```

### 5. Test Edge Cases

Test NAF with:
- Empty knowledge base (everything should fail)
- Contradictory facts (both `p(a)` and `~p(a)`)
- Unbound variables (wildcards)
- Deeply nested rules

### 6. Monitor Stratification Warnings

Always review stratification warnings - they indicate potential semantic issues.

## Examples

### Example 1: Access Control

```prolog
@model FHRR(dim=1024, seed=42);

// Users and resources
fact user(alice).
fact user(bob).
fact user(carol).
fact resource(file1).
fact resource(file2).

// Access permissions
fact has_permission(alice, file1).
fact has_permission(bob, file2).

// Explicit denials (classical negation)
fact ~has_permission(carol, file1).

// Banned users
fact banned(carol).

// Rules with NAF
rule can_access(U, R) :-
    user(U),
    resource(R),
    has_permission(U, R),
    not banned(U).

// Queries
query can_access(alice, file1)?   // Yes (has permission, not banned)
query can_access(bob, file2)?     // Yes (has permission, not banned)
query can_access(carol, file1)?   // No (banned)
query can_access(alice, file2)?   // No (no permission)
```

### Example 2: Eligibility System

```prolog
@model FHRR(dim=1024, seed=42);

// Applicants
fact applicant(alice).
fact applicant(bob).
fact applicant(carol).
fact applicant(dave).

// Requirements met
fact has_degree(alice).
fact has_degree(bob).
fact has_degree(carol).

fact has_experience(alice).
fact has_experience(dave).

// Disqualifications
fact disqualified(dave).

// Classical negation: explicit rejections
fact ~has_degree(dave).

// NAF rules: eligibility logic
rule meets_requirements(X) :-
    applicant(X),
    has_degree(X),
    has_experience(X).

rule eligible(X) :-
    meets_requirements(X),
    not disqualified(X).

// Queries
query eligible(X)?                  // Returns: alice
query meets_requirements(X)?        // Returns: alice
query ~has_degree(dave)?            // Returns: true (explicit negative fact)
```

### Example 3: Route Planning with Blockages

```prolog
@model FHRR(dim=1024, seed=42);

// Road network
fact road(a, b).
fact road(b, c).
fact road(c, d).
fact road(a, e).
fact road(e, d).

// Blocked roads
fact blocked(b, c).

// Available routes (not blocked)
rule available(X, Y) :-
    road(X, Y),
    not blocked(X, Y).

// Reachable in one step
rule reachable_1(X, Y) :-
    available(X, Y).

// Reachable via intermediate point
rule reachable(X, Y) :-
    available(X, Y).

rule reachable(X, Z) :-
    available(X, Y),
    reachable(Y, Z).

// Queries
query available(X, Y)?      // All non-blocked roads
query reachable(a, d)?      // Route from a to d avoiding blocked roads
```

### Example 4: Trust Network

```prolog
@model FHRR(dim=1024, seed=42);

// People
fact person(alice).
fact person(bob).
fact person(carol).
fact person(dave).

// Trust relationships
fact trusts(alice, bob).
fact trusts(bob, carol).
fact trusts(alice, carol).

// Explicit distrust (classical negation)
fact ~trusts(alice, dave).
fact ~trusts(carol, dave).

// Suspicious activity
fact suspicious(dave).

// Rules
rule trusted_friend(X, Y) :-
    person(X),
    person(Y),
    trusts(X, Y),
    not suspicious(Y).

rule trustworthy(X) :-
    person(X),
    not suspicious(X).

// Indirect trust
rule trusts_indirectly(X, Z) :-
    trusts(X, Y),
    trusts(Y, Z),
    not suspicious(Z).

// Queries
query trusted_friend(alice, X)?      // Bob, Carol (not Dave - suspicious)
query trustworthy(X)?                // Alice, Bob, Carol (not Dave)
query trusts_indirectly(alice, X)?   // Carol (via Bob)
```

## Troubleshooting

### Issue 1: NAF Not Working as Expected

**Problem**: NAF literal always returns False

**Diagnosis**:
- Check if facts exist for the predicate
- Verify variable bindings
- Check for typos in predicate names

**Solution**:
```python
# Debug: Check what facts exist
facts = engine.kb.get_facts("enemy")
print(f"Enemy facts: {facts}")
```

### Issue 2: Stratification Warning

**Problem**: Warning about non-stratified program

**Diagnosis**:
```prolog
// Find the cycle
rule p(X) :- not q(X).
rule q(X) :- not p(X).   // Creates negative cycle
```

**Solution**: Restructure to break cycle:
```prolog
// Option 1: Use classical negation
fact ~q(alice).
rule p(alice).

// Option 2: Rewrite logic
rule p(X) :- base_fact(X), not excluded(X).
```

### Issue 3: Unexpected Results with Contradictions

**Problem**: Both `p(a)` and `~p(a)` exist, confusing results

**Solution**: Use consistency checker:
```python
from vsar.reasoning.consistency import check_consistency

result = check_consistency(engine.kb)
if not result.is_consistent:
    print("Contradictions found:")
    for c in result.contradictions:
        print(f"  {c.predicate}{c.args}")
```

### Issue 4: NAF Too Slow

**Problem**: NAF evaluation is slow with large knowledge bases

**Diagnosis**: NAF checks all facts matching the predicate

**Solution**:
- Use predicate partitioning (automatically done)
- Reduce beam width for join operations
- Consider caching NAF results (future optimization)

### Issue 5: Closed-World Assumption Issues

**Problem**: NAF assumes false when you meant unknown

**Solution**: Use classical negation for explicit unknowns:
```prolog
// Instead of relying on NAF
rule safe(X) :- person(X), not enemy(X, Y).

// Add explicit negative facts for known non-enemies
fact ~enemy(alice, bob).
fact ~enemy(alice, carol).

// Now NAF has explicit data to work with
```

## Summary

### Key Takeaways

1. **Two forms of negation**: Classical (`~`) for explicit falsity, NAF (`not`) for absence of proof
2. **Different locations**: Classical in facts, NAF in rule bodies
3. **Complementary**: Use both together for expressive reasoning
4. **Stratification matters**: Keep programs stratified for predictable semantics
5. **Test thoroughly**: Negation can have subtle edge cases

### Quick Reference

```prolog
// Classical negation (facts)
fact ~criminal(alice).

// Negation-as-failure (rules)
rule safe(X) :- person(X), not enemy(X, Y).

// Query negative facts
query ~criminal(X)?

// Wildcards in NAF
rule eligible(X) :- applicant(X), not disqualified(X).

// Stratification
// ✅ Safe: depends on lower stratum
rule derived(X) :- base(X), not excluded(X).

// ❌ Unsafe: negative cycle
rule p(X) :- not q(X).
rule q(X) :- not p(X).
```

### Next Steps

- Read the [Language Reference](../language-reference.md) for complete syntax
- See [Syntax Cheatsheet](../syntax-cheatsheet.md) for quick examples
- Try the [negation example](../../examples/07_negation.vsar)
- Explore [stratification tests](../../tests/negation/test_stratification.py) for edge cases

---

**Need help?** Check the [VSAR documentation](../index.md) or open an issue on GitHub.
