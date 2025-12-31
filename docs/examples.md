# Example Programs

VSAR includes 6 comprehensive example programs demonstrating different reasoning patterns. All examples are available in the [`examples/`](https://github.com/vasanthsarathy/vsar/tree/main/examples) directory.

## Quick Links

- **[01_basic_rules.vsar](#example-1-basic-rules)** - Simple rule derivation
- **[02_family_tree.vsar](#example-2-family-tree)** - Multi-hop grandparent inference
- **[03_transitive_closure.vsar](#example-3-transitive-closure)** - Recursive ancestor rules
- **[04_organizational_hierarchy.vsar](#example-4-organizational-hierarchy)** - Manager chains
- **[05_knowledge_graph.vsar](#example-5-knowledge-graph)** - Multi-relation connections
- **[06_academic_network.vsar](#example-6-academic-network)** - Complex multi-rule interactions

## Example 1: Basic Rules

**File:** `examples/01_basic_rules.vsar`

**Demonstrates:** Simple rule derivation

```prolog
// Configuration
@model FHRR(dim=512, seed=42);
@beam 50;
@novelty 0.95;

// Base facts: People
fact person(alice).
fact person(bob).
fact person(carol).

// Rule: Humans are persons
rule human(X) :- person(X).

// Query: Who are the humans?
query human(?)?
```

**Run it:**
```bash
vsar run examples/01_basic_rules.vsar
```

**Key Concepts:**
- Single-body rules
- Simple variable substitution
- Basic rule application

## Example 2: Family Tree

**File:** `examples/02_family_tree.vsar`

**Demonstrates:** Multi-hop inference with grandparent rules

```prolog
@model FHRR(dim=1024, seed=42);
@beam 50;
@novelty 0.95;

// Base facts: Parent relationships
fact parent(alice, bob).
fact parent(alice, carol).
fact parent(bob, dave).
fact parent(bob, eve).
fact parent(carol, frank).

// Rule: Grandparent relationship
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

// Queries
query grandparent(alice, ?)?      // Alice's grandchildren
query grandparent(?, dave)?       // Dave's grandparents
query parent(alice, ?)?           // Alice's children (base facts)
```

**Key Concepts:**
- Multi-body rules (joins)
- Variable binding across atoms
- Deriving new relationships from base facts

## Example 3: Transitive Closure

**File:** `examples/03_transitive_closure.vsar`

**Demonstrates:** Recursive rules and multi-hop inference

```prolog
@model FHRR(dim=1024, seed=42);
@beam 50;
@novelty 0.95;

// Base facts: Parent relationships (3 generations)
fact parent(alice, bob).
fact parent(bob, carol).
fact parent(carol, dave).
fact parent(dave, eve).

// Rules: Ancestor relationship (transitive closure)
rule ancestor(X, Y) :- parent(X, Y).                    // Base case
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).    // Recursive case

// Queries
query ancestor(alice, ?)?   // All of Alice's descendants
query ancestor(?, eve)?     // All of Eve's ancestors
```

**Key Concepts:**
- Recursive rules
- Transitive closure
- Multi-generation inference
- Fixpoint detection

## Example 4: Organizational Hierarchy

**File:** `examples/04_organizational_hierarchy.vsar`

**Demonstrates:** Manager chains and reporting relationships

```prolog
@model FHRR(dim=512, seed=100);
@beam 50;
@novelty 0.95;

// Base facts: Management structure
fact manages(ceo, vp_eng).
fact manages(ceo, vp_sales).
fact manages(vp_eng, dir_backend).
fact manages(vp_eng, dir_frontend).
fact manages(dir_backend, mgr_api).
fact manages(dir_frontend, mgr_ui).
fact manages(mgr_api, dev_alice).
fact manages(mgr_api, dev_bob).
fact manages(mgr_ui, dev_carol).

// Rules: Reports-to relationship (transitive)
rule reports_to(X, Y) :- manages(Y, X).                      // Base case
rule reports_to(X, Z) :- manages(Y, X), reports_to(Y, Z).    // Recursive case

// Queries
query reports_to(dev_alice, ?)?    // Who does Alice report to (all levels)?
query reports_to(?, ceo)?          // Who reports to the CEO (all levels)?
```

**Key Concepts:**
- Real-world hierarchies
- Reverse relationships (manages → reports_to)
- Multi-level transitive inference

## Example 5: Knowledge Graph

**File:** `examples/05_knowledge_graph.vsar`

**Demonstrates:** Multiple relation types combined via rules

```prolog
@model FHRR(dim=512, seed=50);
@beam 50;
@novelty 0.95;

// Base facts: Different types of relationships
fact knows(alice, bob).
fact knows(bob, carol).
fact knows(carol, dave).
fact works_with(alice, eve).
fact works_with(eve, dave).
fact works_with(bob, frank).

// Rules: General connection (combines both relationship types)
rule connected(X, Y) :- knows(X, Y).                       // Via knows
rule connected(X, Y) :- works_with(X, Y).                  // Via works_with
rule connected(X, Z) :- connected(X, Y), connected(Y, Z).  // Transitive

// Queries
query connected(alice, ?)?   // Who is Alice connected to (all paths)?
query knows(alice, ?)?       // Who does Alice know (direct)?
query works_with(alice, ?)?  // Who does Alice work with (direct)?
```

**Key Concepts:**
- Multiple predicates
- Rule disjunction (multiple rules with same head)
- Heterogeneous graph reasoning

## Example 6: Academic Network

**File:** `examples/06_academic_network.vsar`

**Demonstrates:** Complex multi-rule scenario with different patterns

```prolog
@model FHRR(dim=1024, seed=42);
@beam 100;
@novelty 0.95;

// Base facts: Academic relationships
fact advises(prof_smith, phd_alice).
fact advises(phd_alice, ms_bob).
fact advises(prof_jones, phd_carol).
fact advises(phd_carol, ms_dave).
fact coauthor(prof_smith, prof_jones).
fact coauthor(phd_alice, phd_carol).

// Rules: Complex academic network

// Academic lineage (transitive advising)
rule academic_ancestor(X, Y) :- advises(X, Y).
rule academic_ancestor(X, Z) :- advises(X, Y), academic_ancestor(Y, Z).

// Symmetric coauthorship
rule coauthor(Y, X) :- coauthor(X, Y).

// Collaboration network (transitive)
rule collaborator(X, Y) :- coauthor(X, Y).
rule collaborator(X, Z) :- coauthor(X, Y), collaborator(Y, Z).

// Academic siblings (share same advisor)
rule academic_sibling(X, Y) :- advises(Z, X), advises(Z, Y).

// Queries
query academic_ancestor(prof_smith, ?)?   // Prof Smith's academic descendants
query collaborator(prof_smith, ?)?        // Prof Smith's collaboration network
query academic_sibling(phd_alice, ?)?     // PhD Alice's academic siblings
```

**Key Concepts:**
- Multiple rule types interacting
- Symmetric relationships
- Multiple derivation paths
- Complex network analysis

## Running Examples

### From Command Line

```bash
# Run a specific example
vsar run examples/02_family_tree.vsar

# Run with limited results
vsar run examples/03_transitive_closure.vsar --k 5

# Run with trace output
vsar run examples/04_organizational_hierarchy.vsar --trace

# Run with JSON output
vsar run examples/05_knowledge_graph.vsar --json
```

### From Python

```python
from vsar.language.loader import ProgramLoader
from vsar.semantics.engine import VSAREngine

# Load program
loader = ProgramLoader()
program = loader.load_file("examples/02_family_tree.vsar")

# Create engine from directives
engine = VSAREngine(program.directives)

# Insert facts
for fact in program.facts:
    engine.insert_fact(fact)

# Execute queries with rules
for query in program.queries:
    result = engine.query(query, rules=program.rules, k=10)
    print(f"Query: {query.predicate}({', '.join(str(a) for a in query.args)})")
    for entity, score in result.results:
        print(f"  {entity}: {score:.4f}")
```

## Example Patterns

### Pattern 1: Base + Recursive Rules

Used for transitive closure (ancestors, paths, reachability):

```prolog
rule derived(X, Y) :- base(X, Y).                    // Base case
rule derived(X, Z) :- base(X, Y), derived(Y, Z).     // Recursive case
```

### Pattern 2: Multi-Body Derivation

Used for derived relationships (grandparent, uncle):

```prolog
rule derived(X, Z) :- relation1(X, Y), relation2(Y, Z).
```

### Pattern 3: Symmetric Rules

Used for symmetric relationships (coauthor, friend):

```prolog
rule symmetric(Y, X) :- symmetric(X, Y).
```

### Pattern 4: Combining Multiple Sources

Used for unified relationships (connected, related):

```prolog
rule unified(X, Y) :- source1(X, Y).
rule unified(X, Y) :- source2(X, Y).
rule unified(X, Z) :- unified(X, Y), unified(Y, Z).  // Make transitive
```

## Creating Your Own Examples

### Template

```prolog
// 1. Configuration
@model FHRR(dim=1024, seed=42);
@beam 50;
@novelty 0.95;

// 2. Base facts
fact predicate(arg1, arg2).
// ... more facts ...

// 3. Rules
rule derived(X, Z) :- base(X, Y), base(Y, Z).
// ... more rules ...

// 4. Queries
query derived(alice, ?)?
// ... more queries ...
```

### Tips

1. **Start simple** - Begin with a few facts and one rule
2. **Test incrementally** - Add rules one at a time
3. **Check fixpoint** - Verify chaining completes
4. **Adjust beam width** - Increase if missing results
5. **Monitor novelty** - Lower threshold if duplicates
6. **Use traces** - Debug with `--trace` flag

## Next Steps

- **[Tutorials](tutorials/basic-queries.md)** - Step-by-step guides
- **[Rules & Chaining Guide](guides/rules-and-chaining.md)** - Deep dive into reasoning
- **[Language Reference](language-reference.md)** - Complete syntax
- **[Python API](guides/python-api.md)** - Programmatic usage
