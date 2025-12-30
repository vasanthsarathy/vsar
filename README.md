# VSAR: VSA-grounded Reasoning

VSAR (VSAX Reasoner) is a VSA-grounded reasoning system that provides fast approximate querying over large knowledge bases using hypervector algebra. Built on the [VSAX library](https://vsarathy.com/vsax/) for GPU-accelerated VSA operations.

## Key Features

- **Fast approximate querying**: Query 10^6+ facts with subsymbolic retrieval
- **VSARL language**: Declarative syntax for facts, queries, and rules
- **Interactive REPL**: Load files and query interactively
- **CLI interface**: Simple commands for ingestion, querying, and export
- **Multiple formats**: Load facts from CSV, JSONL, or VSAR files
- **Trace layer**: Explanation DAG for debugging and transparency
- **Deterministic results**: Reproducible outputs with fixed seeds
- **HDF5 persistence**: Save and load knowledge bases
- **Comprehensive testing**: 295 tests with 98.6% coverage

## Quick Start

### Installation

**Option 1: Install from PyPI (recommended for users)**

```bash
pip install vsar

# Verify installation
vsar --help
```

**Option 2: Development install with uv**

```bash
# Install uv
pip install uv

# Clone and install
git clone https://github.com/vasanthsarathy/vsar.git
cd vsar
uv sync

# For development, use uv run
uv run vsar --help
```

### Hello World - CLI

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
# After pip install vsar
vsar run family.vsar

# Or during development with uv
uv run vsar run family.vsar
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

### Interactive REPL

Start an interactive session to load files and query on the fly:

```bash
vsar repl
```

Example session:

```
VSAR Interactive REPL
Type 'help' for commands, 'exit' to quit

> load family.vsar
Loaded family.vsar
Inserted 4 facts

> query parent(alice, X)?
┌─────────────────────────┐
│ Query: parent(alice, X) │
├────────┬────────────────┤
│ Entity │ Score          │
├────────┼────────────────┤
│ bob    │ 0.9234         │
│ carol  │ 0.9156         │
└────────┴────────────────┘

> query parent(X, dave)?
┌───────────────────────┐
│ Query: parent(X, dave)│
├────────┬──────────────┤
│ Entity │ Score        │
├────────┼──────────────┤
│ bob    │ 0.8876       │
└────────┴──────────────┘

> stats
Knowledge Base Statistics
Total Facts: 4
Predicates: parent (4 facts)

> exit
Goodbye!
```

### CLI Commands

#### Ingest Facts

```bash
# From CSV (predicate in first column)
vsar ingest facts.csv --kb family.h5

# From CSV (all rows same predicate)
vsar ingest parents.csv --predicate parent --kb family.h5

# From JSONL
vsar ingest facts.jsonl --kb family.h5
```

#### Query and Export

```bash
# Export KB to JSON
vsar export family.h5 --format json --output facts.json

# Export to JSONL
vsar export family.h5 --format jsonl --output facts.jsonl

# Inspect KB statistics
vsar inspect family.h5
```

#### Advanced Options

```bash
# JSON output for scripting
vsar run program.vsar --json

# Show trace DAG
vsar run program.vsar --trace

# Limit results per query
vsar run program.vsar --k 10
```

## VSARL Language

### Directives

Configure the reasoning engine:

```prolog
// Model configuration
@model FHRR(dim=8192, seed=42);    // FHRR backend, 8192 dimensions
@model MAP(dim=4096, seed=100);     // MAP backend (alternative)

// Retrieval parameters
@threshold(value=0.22);             // Similarity threshold
@beam(width=50);                    // Beam width (Phase 2)
```

### Facts

Ground atoms (all arguments are constants):

```prolog
fact parent(alice, bob).
fact parent(bob, carol).
fact lives_in(alice, boston).
fact transfer(alice, bob, money).   // Ternary fact
fact person(alice).                  // Unary fact
```

### Queries

Single-atom queries with one variable (Phase 1):

```prolog
query parent(alice, X)?         // Find children of alice
query parent(X, carol)?         // Find parents of carol
query lives_in(X, boston)?      // Who lives in boston?
query transfer(alice, X, money)? // Alice transferred money to X
```

**Phase 1 Limitation**: Only single-variable, single-atom queries supported. Conjunctive queries coming in Phase 2.

### Comments

```prolog
// Single-line comment

/* Multi-line
   comment */
```

## File Formats

### CSV Format

**With predicate column** (first column = predicate):
```csv
parent,alice,bob
parent,bob,carol
lives_in,alice,boston
```

**Without predicate** (use `--predicate` flag):
```csv
alice,bob
bob,carol
```

### JSONL Format

One fact per line:
```jsonl
{"predicate": "parent", "args": ["alice", "bob"]}
{"predicate": "parent", "args": ["bob", "carol"]}
{"predicate": "lives_in", "args": ["alice", "boston"]}
```

### VSAR Format

Native `.vsar` program files (see VSARL Language above).

## Python API

### High-Level API (Recommended)

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.language.loader import load_facts
from vsar.semantics.engine import VSAREngine

# Create engine from directives
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
]
engine = VSAREngine(directives)

# Load and insert facts
facts = load_facts("facts.csv")
for fact in facts:
    engine.insert_fact(fact)

# Execute query
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=5)

for entity, score in result.results:
    print(f"{entity}: {score:.4f}")

# Inspect trace
trace = engine.trace.get_dag()
for event in trace:
    print(f"{event.type}: {event.payload}")

# Save KB
engine.save_kb("family.h5")
```

### Low-Level API (Phase 0 Foundation)

```python
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.encoding.roles import RoleVectorManager
from vsar.kb.store import KnowledgeBase
from vsar.retrieval.query import Retriever

# Create VSA system
backend = FHRRBackend(dim=512, seed=42)
registry = SymbolRegistry(backend, seed=42)
encoder = VSAEncoder(backend, registry, seed=42)
kb = KnowledgeBase(backend)
role_manager = RoleVectorManager(backend, seed=42)
retriever = Retriever(backend, registry, kb, encoder, role_manager)

# Insert facts
atom_vec = encoder.encode_atom("parent", ["alice", "bob"])
kb.insert("parent", atom_vec, ("alice", "bob"))

# Query: parent(alice, X)
results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)
print(results)  # [('bob', 0.85), ...]
```

## Architecture

VSAR uses a layered architecture:

### Phase 0 Layers (Foundation)

- **Kernel** (`vsar.kernel`): VSA operations (FHRR/MAP backends via VSAX)
- **Symbols** (`vsar.symbols`): Typed symbol spaces (E, R, A, C, T, S) with basis management
- **Encoding** (`vsar.encoding`): Role-filler binding for atoms (predicate + arguments)
- **KB** (`vsar.kb`): Predicate-partitioned storage with HDF5 persistence
- **Retrieval** (`vsar.retrieval`): Unbinding, cleanup, top-k similarity search

### Phase 1 Layers (Language & CLI)

- **Language** (`vsar.language`): VSARL parser (Lark), AST, loaders (CSV/JSONL/VSAR)
- **Semantics** (`vsar.semantics`): VSAREngine orchestrating all layers
- **Trace** (`vsar.trace`): Explanation DAG for transparency
- **CLI** (`vsar.cli`): Typer-based commands with Rich formatting

See [docs/architecture.md](docs/architecture.md) for complete details.

## Project Status

### ✅ Phase 0 (Foundation) - COMPLETE

- ✅ Kernel backend (FHRR VSA via VSAX)
- ✅ Symbol space management (6 typed spaces)
- ✅ Atom encoding (role-filler binding)
- ✅ KB storage (predicate-partitioned bundles)
- ✅ Retrieval primitive (unbind → cleanup)
- ✅ HDF5 persistence (KB + basis)
- ✅ Published to PyPI (v0.1.0)

### ✅ Phase 1 (Language & CLI) - COMPLETE

- ✅ VSARL parser (facts, queries, directives)
- ✅ Facts ingestion (CSV/JSONL/VSAR)
- ✅ Program execution engine
- ✅ Trace layer (explanation DAG)
- ✅ CLI interface (run, ingest, export, inspect)
- ✅ 281 tests, 98.5% coverage

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

## Examples

### Example 1: Family Tree

```prolog
@model FHRR(dim=8192, seed=42);

fact parent(alice, bob).
fact parent(bob, carol).
fact parent(carol, dave).

query parent(alice, X)?     // Returns: bob (0.92)
query parent(X, carol)?     // Returns: bob (0.88)
```

### Example 2: Knowledge Graph

```prolog
@model FHRR(dim=8192, seed=42);
@threshold(value=0.25);

fact lives_in(alice, boston).
fact lives_in(bob, cambridge).
fact works_at(alice, mit).
fact works_at(bob, harvard).

query lives_in(X, boston)?    // Returns: alice
query works_at(alice, X)?     // Returns: mit
```

### Example 3: Large-Scale Ingestion

```bash
# Ingest 1M facts from CSV
vsar ingest large_dataset.csv \
  --kb large.h5 \
  --dim 8192 \
  --seed 42

# Query the KB
vsar run queries.vsar --k 10
```

## Performance

**Approximate query performance** (Phase 1):
- 10^3 facts: <100ms per query
- 10^4 facts: <200ms per query
- 10^5 facts: <500ms per query
- 10^6 facts: <1s per query

*Measured on AMD EPYC 7742 CPU with dim=8192*

## Testing

VSAR has comprehensive test coverage:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=vsar --cov-report=html

# Run specific suites
uv run pytest tests/unit/           # Unit tests
uv run pytest tests/integration/    # Integration tests
```

**Test statistics:**
- **281 tests** (all passing)
- **98.5% coverage**
- Unit tests: 261
- Integration tests: 20

## Development

```bash
# Install development dependencies
uv sync --all-groups

# Run formatters
uv run black .
uv run ruff check . --fix

# Type checking
uv run mypy src/vsar

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files

# Build documentation
cd docs && uv run mkdocs serve
```

## Documentation

- [Architecture Overview](docs/architecture.md) - System design and layer details
- [Getting Started](docs/getting-started.md) - Tutorials and examples
- [API Reference](docs/api/) - Complete API documentation
- [CLAUDE.md](CLAUDE.md) - Developer workflow guide

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use VSAR in your research, please cite:

```bibtex
@software{vsar2025,
  title = {VSAR: VSA-grounded Reasoning},
  author = {VSAR Contributors},
  year = {2025},
  url = {https://github.com/your-org/vsar}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on [VSAX](https://vsarathy.com/vsax/) for VSA operations
- Inspired by Datalog and logic programming systems
- Uses [Lark](https://github.com/lark-parser/lark) for parsing
- CLI powered by [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)
