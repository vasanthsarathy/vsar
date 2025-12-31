# VSAR Documentation

**VSAR (VSAX Reasoner)** is a VSA-grounded reasoning system that combines **Datalog-style logic programming** with **approximate vector matching**. Built on the [VSAX library](https://vsarathy.com/vsax/) for GPU-accelerated hypervector operations, VSAR enables fast approximate reasoning over large knowledge bases with explainable results.

!!! tip "Think of it as"
    "Datalog meets vector similarity search" - a foundation for approximate deductive reasoning at scale.

## Key Features

### Deductive Reasoning
- **Horn clause rules** - Full support for `head :- body1, body2, ...` syntax
- **Forward chaining** - Iterative rule application with fixpoint detection
- **Transitive closure** - Multi-hop inference (arbitrary depth)
- **Semi-naive evaluation** - Optimized chaining that avoids redundant work

### Approximate Matching
- **VSA-based similarity** - Fuzzy matching with confidence scores instead of exact symbolic matching
- **Graceful degradation** - Works with noisy data and typos
- **Beam search joins** - Prevents combinatorial explosion in multi-body rules
- **Novelty detection** - Prevents duplicate derivations via similarity threshold

### Performance & Scale
- **Fast approximate querying** - Query 10^6+ facts with subsymbolic retrieval
- **Vectorized operations** - GPU-ready via JAX backend
- **Predicate partitioning** - Efficient KB organization
- **HDF5 persistence** - Save and load knowledge bases

### Developer Experience
- **VSARL language** - Declarative syntax for facts, queries, and rules
- **Interactive REPL** - Load files and query interactively
- **CLI interface** - Simple commands for ingestion, querying, and export
- **Full traceability** - Explanation DAG for debugging and transparency
- **Comprehensive testing** - 392 tests with 97.56% coverage

## Quick Start

### Installation

```bash
# Install from PyPI
pip install vsar

# Verify installation
vsar --help
```

### Hello World - Facts & Queries

Create a simple VSAR program `family.vsar`:

```prolog
@model FHRR(dim=1024, seed=42);

fact parent(alice, bob).
fact parent(bob, carol).
fact parent(carol, dave).

query parent(alice, X)?
query parent(X, carol)?
```

Run it:

```bash
vsar run family.vsar
```

Output:

```
Inserted 3 facts

┌─────────────────────────┐
│ Query: parent(alice, X) │
├────────┬────────────────┤
│ Entity │ Score          │
├────────┼────────────────┤
│ bob    │ 0.9234         │
└────────┴────────────────┘
```

### Hello World - Rules & Reasoning

Now let's add some rules for deductive reasoning:

```prolog
@model FHRR(dim=1024, seed=42);
@beam 50;
@novelty 0.95;

// Base facts
fact parent(alice, bob).
fact parent(bob, carol).
fact parent(carol, dave).

// Rule: Derive grandparent relationship
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

// Rule: Transitive closure for ancestors
rule ancestor(X, Y) :- parent(X, Y).
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

// Queries
query grandparent(alice, X)?
query ancestor(alice, X)?
```

Run it:

```bash
vsar run reasoning.vsar
```

Output:

```
Inserted 3 facts

Applied 3 rules in 2 iterations
Derived 5 new facts
Fixpoint reached: true

┌──────────────────────────────┐
│ Query: grandparent(alice, X) │
├────────┬─────────────────────┤
│ Entity │ Score               │
├────────┼─────────────────────┤
│ carol  │ 0.8456              │
└────────┴─────────────────────┘

┌───────────────────────────┐
│ Query: ancestor(alice, X) │
├────────┬──────────────────┤
│ Entity │ Score            │
├────────┼──────────────────┤
│ bob    │ 0.9234           │
│ carol  │ 0.8876           │
│ dave   │ 0.8123           │
└────────┴──────────────────┘
```

## Documentation Structure

### Getting Started
- **[Installation](getting-started.md)** - Installation and first steps
- **[Quick Start](guides/quick-start.md)** - Get up and running in 5 minutes
- **[Your First Program](guides/first-program.md)** - Step-by-step tutorial

### Tutorials
- **[Basic Facts & Queries](tutorials/basic-queries.md)** - Learn the basics
- **[Rules & Reasoning](tutorials/rules-and-reasoning.md)** - Deductive inference
- **[Transitive Closure](tutorials/transitive-closure.md)** - Multi-hop reasoning
- **[Knowledge Graphs](tutorials/knowledge-graphs.md)** - Complex relationships
- **[Advanced Patterns](tutorials/advanced-patterns.md)** - Expert techniques

### User Guides
- **[Basic Usage](guides/basic-usage.md)** - Facts, queries, and programs
- **[Rules & Chaining](guides/rules-and-chaining.md)** - Horn clauses and forward chaining
- **[File Formats](guides/file-formats.md)** - CSV, JSONL, and VSAR formats
- **[KB Management](guides/kb-management.md)** - Persistence and export
- **[Python API](guides/python-api.md)** - Using VSAR programmatically
- **[Performance Tuning](guides/performance.md)** - Optimization tips

### Reference
- **[CLI Reference](cli-reference.md)** - Complete CLI command reference
- **[Language Reference](language-reference.md)** - VSARL syntax guide
- **[Capabilities & Limitations](capabilities.md)** - What VSAR can and cannot do
- **[Architecture](architecture.md)** - System design and internals
- **[API Reference](api/index.md)** - Python API documentation

## Project Status

### ✅ Phase 0: Foundation (Complete)
- VSA kernel (FHRR, MAP backends)
- Symbol registry and encoding
- KB storage with predicate partitioning
- Basic retrieval with similarity search
- HDF5 persistence

### ✅ Phase 1: Ground KB + Queries (Complete)
- VSARL language parser
- Facts ingestion (CSV/JSONL/VSAR)
- Single-variable queries
- CLI interface and REPL
- Trace collection

### ✅ Phase 2: Horn Rules + Chaining (Complete)
- Horn clause rules
- Variable substitution and unification
- Beam search joins
- Forward chaining with fixpoint detection
- Semi-naive evaluation
- Novelty detection
- Query with automatic rule application

**Stats:** 392 tests passing, 97.56% coverage

### 🔜 Phase 3: Advanced Features (Planned)
- Multi-variable queries (`parent(?, ?)?`)
- Stratified negation
- Aggregation (count, sum, max)
- Backward chaining
- Magic sets optimization

### 🔜 Phase 4: Scale & Performance (Planned)
- Incremental maintenance
- Query planning and optimization
- Parallel execution
- GPU acceleration

See [Capabilities & Limitations](capabilities.md) for detailed status.

## Performance

**Approximate query performance** (Phase 2, with rules):

| Facts | Query Time | Chaining Time (10 rules) |
|-------|------------|--------------------------|
| 10^3  | <50ms      | <200ms                   |
| 10^4  | <100ms     | <500ms                   |
| 10^5  | <300ms     | <2s                      |
| 10^6  | <800ms     | <10s                     |

*Measured on AMD EPYC 7742 CPU with dim=1024, beam=50*

## Use Cases

**Best suited for:**

1. Knowledge graph reasoning with noise tolerance
2. Transitive closure queries (org hierarchies, supply chains)
3. Multi-hop reasoning (family trees, social networks)
4. Explainable AI (need provenance and similarity scores)
5. Large-scale approximate reasoning (vectorized operations)

**Not yet suitable for:**

1. Complex logical puzzles requiring negation
2. Planning problems (need backward chaining)
3. Ontology reasoning (need DL features)
4. Answer set programming tasks

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/vasanthsarathy/vsar/issues)
- **Discussions**: [Ask questions](https://github.com/vasanthsarathy/vsar/discussions)
- **Documentation**: You're reading it!
- **Examples**: [Browse example programs](examples.md)

## Contributing

Contributions welcome! See our [GitHub repository](https://github.com/vasanthsarathy/vsar) for guidelines.

## License

VSAR is released under the MIT License. See [LICENSE](https://github.com/vasanthsarathy/vsar/blob/main/LICENSE) for details.

## Citation

If you use VSAR in your research, please cite:

```bibtex
@software{vsar2025,
  title = {VSAR: VSA-grounded Reasoning},
  author = {Sarathy, Vasanth},
  year = {2025},
  url = {https://github.com/vasanthsarathy/vsar},
  version = {0.3.0}
}
```
