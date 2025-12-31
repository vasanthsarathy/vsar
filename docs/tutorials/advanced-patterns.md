# Tutorial: Advanced Patterns

Advanced rule patterns and techniques for expert users.

## Pattern 1: Multiple Interacting Rules

```prolog
// Academic network with multiple rule types
fact advises(prof, phd_student).
fact advises(phd_student, ms_student).
fact coauthor(prof, colleague).

// Academic lineage (transitive)
rule academic_ancestor(X, Y) :- advises(X, Y).
rule academic_ancestor(X, Z) :- advises(X, Y), academic_ancestor(Y, Z).

// Symmetric coauthorship
rule coauthor(Y, X) :- coauthor(X, Y).

// Collaboration network
rule collaborator(X, Y) :- coauthor(X, Y).
rule collaborator(X, Z) :- coauthor(X, Y), collaborator(Y, Z).

// Academic siblings (same advisor)
rule academic_sibling(X, Y) :- advises(Z, X), advises(Z, Y).
```

See [Example 6](../examples.md#example-6-academic-network) for full program.

## Pattern 2: Conditional Derivation

```prolog
// Derive relationships based on multiple conditions
rule qualified(X) :- has_degree(X), has_experience(X, Y).
rule senior(X) :- employee(X), tenure(X, T), greater_than(T, 10).
```

**Note:** Arithmetic predicates not yet implemented (Phase 3).

## Pattern 3: Combining Rules

```prolog
// Build complex relationships from simple ones
rule family(X, Y) :- parent(X, Y).
rule family(X, Y) :- sibling(X, Y).
rule family(X, Z) :- family(X, Y), family(Y, Z).
```

## Best Practices

1. **Start simple** - One rule at a time
2. **Test incrementally** - Verify each rule works
3. **Use traces** - Debug with `--trace` flag
4. **Monitor fixpoint** - Check iterations and derived counts
5. **Tune parameters** - Adjust beam width and novelty threshold

## Next Steps

- **[Examples](../examples.md)** - Complete working programs
- **[Rules & Chaining Guide](../guides/rules-and-chaining.md)** - Comprehensive reference
- **[Performance Guide](../guides/performance.md)** - Optimization
