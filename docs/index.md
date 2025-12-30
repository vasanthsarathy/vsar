# VSAR Documentation

**VSAR (VSAX Reasoner)** is a VSA-grounded reasoning system that provides **fast approximate querying over large knowledge bases** using hypervector algebra. Built on the [VSAX library](https://vsarathy.com/vsax/) for GPU-accelerated VSA operations.

## Key Features

- **Fast approximate querying**: Query 10^6+ facts with subsymbolic retrieval
- **VSARL language**: Declarative syntax for facts, queries, and rules
- **CLI interface**: Simple commands for ingestion, querying, and export
- **Multiple formats**: Load facts from CSV, JSONL, or VSAR files
- **Trace layer**: Explanation DAG for debugging and transparency
- **Deterministic results**: Reproducible outputs with fixed seeds
- **HDF5 persistence**: Save and load knowledge bases
- **Comprehensive testing**: 281 tests with 98.5% coverage

## Quick Start

### Installation

```bash
# Install from PyPI
pip install vsar

# Verify installation
vsar --help
```

### Hello World

Create a simple VSAR program `family.vsar`:

```prolog
@model FHRR(dim=8192, seed=42);
@threshold(value=0.22);

fact parent(alice, bob).
fact parent(alice, carol).
fact parent(bob, dave).
fact parent(carol, eve).

query parent(alice, X)?
query parent(X, dave)?
```

Run it:

```bash
vsar run family.vsar
```

Output:

```
Inserted 4 facts

┌─────────────────────────┐
│ Query: parent(alice, X) │
├────────┬────────────────┤
│ Entity │ Score          │
├────────┼────────────────┤
│ bob    │ 0.9234         │
│ carol  │ 0.9156         │
└────────┴────────────────┘
```

## Documentation Structure

- **[Getting Started](getting-started.md)** - Installation and first steps
- **[CLI Reference](cli-reference.md)** - Complete CLI command reference
- **[Language Reference](language-reference.md)** - VSARL syntax guide
- **[User Guides](guides/basic-usage.md)** - Step-by-step tutorials
- **[Architecture](architecture.md)** - System design and internals
- **[API Reference](api/index.md)** - Python API documentation

## User Guides

- **[Basic Usage](guides/basic-usage.md)** - Facts, queries, and programs
- **[File Formats](guides/file-formats.md)** - CSV, JSONL, and VSAR formats
- **[KB Management](guides/kb-management.md)** - Persistence and export
- **[Python API](guides/python-api.md)** - Using VSAR programmatically

## Project Status

### ✅ Phase 0 (Foundation) - COMPLETE

- ✅ Kernel backend (FHRR VSA via VSAX)
- ✅ Symbol space management (6 typed spaces)
- ✅ Atom encoding (role-filler binding)
- ✅ KB storage (predicate-partitioned bundles)
- ✅ Retrieval primitive (unbind → cleanup)
- ✅ HDF5 persistence (KB + basis)

### ✅ Phase 1 (Language & CLI) - COMPLETE (v0.2.0)

- ✅ VSARL parser (facts, queries, directives)
- ✅ Facts ingestion (CSV/JSONL/VSAR)
- ✅ Program execution engine
- ✅ Trace layer (explanation DAG)
- ✅ CLI interface (run, ingest, export, inspect)
- ✅ 281 tests, 98.5% coverage
- ✅ Published to PyPI (v0.2.0)

### 🔜 Phase 2 (Rules & Chaining)

- Rule definitions (`rule grandparent(X,Z) :- parent(X,Y), parent(Y,Z).`)
- Bounded forward chaining
- Conjunctive queries
- Stratified negation

### 🔜 Phase 3 (Optimizations)

- Indexing strategies
- Query planning
- Parallel execution
- Web interface

## Performance

**Approximate query performance** (Phase 1):

- 10^3 facts: <100ms per query
- 10^4 facts: <200ms per query
- 10^5 facts: <500ms per query
- 10^6 facts: <1s per query

*Measured on AMD EPYC 7742 CPU with dim=8192*

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/vasanthsarathy/vsar/issues)
- **Discussions**: [Ask questions](https://github.com/vasanthsarathy/vsar/discussions)
- **Documentation**: You're reading it!

## License

VSAR is released under the MIT License. See [LICENSE](https://github.com/vasanthsarathy/vsar/blob/main/LICENSE) for details.
