# Capabilities & Limitations

This page provides a comprehensive overview of what VSAR can and cannot do in its current state (v0.3.0).

## Current Capabilities (Phase 0-2)

### ✅ Deductive Reasoning

**Ground Facts:**
- Insert ground facts with arbitrary arity
- Unary facts: `person(alice)`
- Binary facts: `parent(alice, bob)`
- Ternary facts: `transfer(alice, bob, money)`
- N-ary facts supported

**Horn Clause Rules:**
- Full `head :- body1, body2, ...` syntax
- Single-body rules: `human(X) :- person(X)`
- Multi-body rules: `grandparent(X, Z) :- parent(X, Y), parent(Y, Z)`
- Recursive rules: `ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)`
- Multiple interacting rules

**Queries:**
- Single-variable queries: `parent(alice, X)?`
- Bound argument queries: `parent(X, bob)?`
- Query with automatic rule application
- Top-k ranked results with similarity scores

**Forward Chaining:**
- Iterative rule application
- Fixpoint detection (stops when no new facts derived)
- Configurable max iterations
- Semi-naive evaluation optimization
- Tracks derived facts per iteration
- Full provenance and traceability

### ✅ Approximate Reasoning

**Similarity-Based Retrieval:**
- Fuzzy matching with confidence scores
- Graceful degradation under noise (typos, similar names)
- No exact symbolic matching required
- Top-k ranked results

**Score Propagation:**
- Confidence scores for all results
- Score propagation through joins (approximate joint probability)
- Novelty detection via similarity threshold (default 0.95)

**Beam Search:**
- Prevents combinatorial explosion in joins
- Configurable beam width (default 50)
- Keeps top-k candidates at each step

### ✅ Performance Features

**Optimizations:**
- Semi-naive evaluation (only processes new facts each iteration)
- Novelty detection (prevents duplicate derivations)
- Predicate partitioning (separate bundles per predicate)
- Beam search (controls join complexity)

**Scale:**
- Vectorized operations (GPU-ready via JAX)
- HDF5 persistence for large KBs
- Handles 10^6+ facts
- Linear scaling with fact count

**Performance Metrics:**
- 10^3 facts: <50ms query, <200ms chaining (10 rules)
- 10^4 facts: <100ms query, <500ms chaining
- 10^5 facts: <300ms query, <2s chaining
- 10^6 facts: <800ms query, <10s chaining

### ✅ Developer Experience

**Language & Interface:**
- Declarative VSARL language
- Interactive REPL
- CLI interface (run, ingest, export, inspect)
- Multiple file formats (CSV, JSONL, VSAR)

**Traceability:**
- Full trace DAG for all operations
- Event types: query, retrieval, fact_insertion, chaining
- Parent-child relationships
- Payload data for debugging

**Testing & Quality:**
- 392 tests passing (4 skipped)
- 97.56% code coverage
- Fixed seeds for reproducibility
- Comprehensive integration tests

## Current Limitations

### ⏳ Single-Variable Queries Only

**What works:**
```prolog
query parent(alice, X)?      ✅ Find children of alice
query parent(X, bob)?        ✅ Find parents of bob
```

**What doesn't work (yet):**
```prolog
query parent(?, ?)?          ❌ All parent-child pairs
query parent(X, Y)?          ❌ Multi-variable queries
```

**Planned:** Phase 3

### ⏳ No Negation

**Missing features:**
```prolog
fact !enemy(alice, bob).              ❌ Classical negation
rule safe(X) :- person(X), not enemy(X, _).  ❌ Negation-as-failure
```

**Why it matters:** Many real-world scenarios need "absence of evidence"

**Planned:** Phase 3 (stratified negation)

### ⏳ No Aggregation

**Missing features:**
```prolog
count(X) :- parent(_, X).    ❌ Count children
max(Age, Person) :- age(Person, Age).  ❌ Find oldest
sum(Salary) :- employee(_, Salary).    ❌ Sum salaries
```

**Planned:** Phase 3

### ⏳ Forward Chaining Only

**Current:**
- Forward chaining: Derive all facts, then query
- Derives facts that may never be queried

**Missing:**
- Backward chaining: Goal-directed search
- Only derive facts needed for query
- Magic sets optimization

**Planned:** Phase 3

### ⏳ No Advanced Query Features

**Missing:**
```prolog
(parent(X, Y) OR sibling(X, Y))     ❌ Disjunction
X knows Y, optionally X likes Y     ❌ Optional patterns
X →* Y                              ❌ SPARQL-like path queries
```

**Planned:** Phase 4+

## Comparison to Other Reasoners

### vs Prolog

| Feature | VSAR | Prolog |
|---------|------|--------|
| Horn clauses | ✅ | ✅ |
| Forward chaining | ✅ | ❌ (backward only) |
| Backward chaining | ❌ | ✅ |
| Negation-as-failure | ❌ | ✅ |
| Approximate matching | ✅ | ❌ |
| Cut operator | ❌ | ✅ |

### vs Datalog

| Feature | VSAR | Datalog |
|---------|------|---------|
| Horn clauses | ✅ | ✅ |
| Forward chaining | ✅ | ✅ |
| Semi-naive | ✅ | ✅ |
| Stratified negation | ❌ | ✅ |
| Aggregation | ❌ | ✅ |
| Multi-var queries | ❌ | ✅ |
| Approximate matching | ✅ | ❌ |

### vs Answer Set Programming (ASP)

| Feature | VSAR | ASP |
|---------|------|-----|
| Horn clauses | ✅ | ✅ |
| Negation-as-failure | ❌ | ✅ |
| Choice rules | ❌ | ✅ |
| Optimization | ❌ | ✅ |
| Stable models | ❌ | ✅ |
| Approximate matching | ✅ | ❌ |

### vs OWL/DL Reasoners

| Feature | VSAR | OWL/DL |
|---------|------|--------|
| Class hierarchies | ❌ | ✅ |
| Property restrictions | ❌ | ✅ |
| Cardinality | ❌ | ✅ |
| Basic inference | ✅ | ✅ |
| Approximate matching | ✅ | ❌ |

## Use Cases

### ✅ Best Suited For

1. **Knowledge graph reasoning with noise tolerance**
   - Family trees with typos
   - Social networks with fuzzy matching
   - Biological pathways with approximate matching

2. **Transitive closure queries**
   - Organizational hierarchies (manager chains)
   - Supply chain tracking (part-of relationships)
   - Code dependency analysis

3. **Multi-hop reasoning**
   - "Who are Alice's grandchildren?"
   - "Which companies are indirectly owned by X?"
   - "What's the shortest path from A to B?"

4. **Explainable AI applications**
   - Need provenance and traces
   - Want similarity scores
   - Require interpretable derivations

5. **Large-scale approximate reasoning**
   - Vectorized operations (fast on GPU)
   - Graceful degradation under noise
   - No need for exact symbolic matching

### ❌ Not Suitable (Yet)

1. **Complex logical puzzles requiring negation**
   - Sudoku solvers
   - Planning problems with constraints
   - Default reasoning scenarios

2. **Planning problems**
   - Need backward chaining
   - Goal-directed search
   - State space exploration

3. **Ontology reasoning**
   - Need DL features (class hierarchies, property restrictions)
   - OWL/RDF compatibility
   - SPARQL query support

4. **Probabilistic reasoning**
   - Need proper uncertainty quantification
   - Bayesian inference
   - Probabilistic rule weights

5. **Answer set programming tasks**
   - Need stable model semantics
   - Choice rules and optimization
   - Constraint solving

## Future Roadmap

### Phase 3: Advanced Features (Next)
- Multi-variable queries - Essential for real use
- Stratified negation - Needed for 80% of real rules
- Backward chaining - For goal-directed queries
- Aggregation - Count, sum, max, min

### Phase 4: Scale & Performance
- Magic sets optimization - Query-driven forward chaining
- Incremental maintenance - Update KB without recomputation
- Query planning - Optimize execution order
- Parallel execution - Multi-threaded chaining

### Phase 5+: Research Features
- Probabilistic VSA - True uncertainty
- Temporal reasoning - Time-varying facts
- Argumentation - Defeasible reasoning
- DL integration - Ontology support

## Production Readiness

**Research Prototype:** ✅ Ready
**Toy Applications:** ✅ Ready
**Production Systems:** ⏳ Needs Phase 3 features

**Assessment:**
- VSAR is production-ready for research prototypes and exploratory applications
- For production use, you need negation and multi-variable queries (Phase 3)
- Current sweet spot: approximate reasoning over noisy knowledge graphs

## Unique Value Proposition

VSAR is the **only reasoner** that combines:
- Datalog-style forward chaining with fixpoint detection
- VSA-based approximate matching with similarity scores
- Semi-naive evaluation for efficiency
- Vectorized operations (GPU-ready via JAX)
- Full traceability and explainability

**Think of it as:** "Datalog meets vector similarity search" - a foundation for approximate deductive reasoning at scale.

## Technical Limitations

### Similarity-Based Matching
- **Tradeoff:** Approximate matching means no guaranteed exact results
- **Impact:** May return similar but incorrect answers
- **Mitigation:** Use high similarity thresholds, verify critical results

### Beam Search Limitations
- **Tradeoff:** Beam width limits number of candidates explored
- **Impact:** May miss valid derivations if beam too narrow
- **Mitigation:** Increase beam width for critical applications

### Memory Usage
- **Linear scaling:** Memory grows linearly with fact count and dimensionality
- **Typical:** ~5MB per 1000 facts (dim=1024)
- **Large KBs:** May require substantial RAM for millions of facts

### Performance Characteristics
- **Best case:** Few facts, simple rules, narrow beam
- **Worst case:** Many facts, complex recursive rules, wide beam
- **Typical:** Subsecond queries for 10^4-10^5 facts

See [Performance Tuning](guides/performance.md) for optimization tips.
