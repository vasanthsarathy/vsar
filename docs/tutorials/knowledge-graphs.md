# Tutorial: Knowledge Graphs

Learn to model and query complex knowledge graphs with multiple relation types.

## What You'll Learn

- Multi-relation graphs
- Combining predicates
- Heterogeneous reasoning
- Path finding

## Example: Social & Professional Network

```prolog
@model FHRR(dim=1024, seed=42);
@beam 100;
@novelty 0.95;

// Social connections
fact knows(alice, bob).
fact knows(bob, carol).
fact knows(carol, dave).

// Professional connections
fact works_with(alice, eve).
fact works_with(eve, dave).
fact works_with(bob, frank).

// Combine both into unified "connected" relationship
rule connected(X, Y) :- knows(X, Y).
rule connected(X, Y) :- works_with(X, Y).

// Make it transitive
rule connected(X, Z) :- connected(X, Y), connected(Y, Z).

// Queries
query knows(alice, X)?         // Direct social connections
query works_with(alice, X)?    // Direct professional connections
query connected(alice, X)?     // All connections (social + professional + transitive)
```

## Pattern: Symmetric Relationships

```prolog
fact coauthor(prof_smith, prof_jones).
fact coauthor(prof_jones, prof_chen).

// Make coauthorship symmetric
rule coauthor(Y, X) :- coauthor(X, Y).

// Make it transitive for collaboration network
rule collaborator(X, Y) :- coauthor(X, Y).
rule collaborator(X, Z) :- coauthor(X, Y), collaborator(Y, Z).

query collaborator(prof_smith, X)?
```

## Pattern: Multi-Hop Paths

```prolog
// Transportation network
fact flight(boston, newyork).
fact flight(newyork, london).
fact train(boston, providence).
fact train(providence, newyork).

// Unify transportation modes
rule route(X, Y) :- flight(X, Y).
rule route(X, Y) :- train(X, Y).
rule route(X, Z) :- route(X, Y), route(Y, Z).

query route(boston, X)?  // All destinations from Boston
```

## Exercises

See [Examples page](../examples.md#example-5-knowledge-graph) for complete working example.

## Next Steps

- **[Advanced Patterns](advanced-patterns.md)** - Complex scenarios
- **[Examples](../examples.md)** - Full programs
