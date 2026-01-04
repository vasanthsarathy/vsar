# VSAR: VSA-grounded Reasoning

[![PyPI version](https://badge.fury.io/py/vsar.svg)](https://badge.fury.io/py/vsar)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-446%20passing-brightgreen.svg)](https://github.com/vasanthsarathy/vsar)
[![Coverage](https://img.shields.io/badge/coverage-97.82%25-brightgreen.svg)](https://github.com/vasanthsarathy/vsar)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

VSAR (VSAX Reasoner) is a **VSA-grounded reasoning system** that combines Datalog-style logic programming with approximate vector matching. Built on [VSAX library](https://vsarathy.com/vsax/) for GPU-accelerated hypervector operations, VSAR enables fast approximate reasoning over large knowledge bases with explainable results.

**Think of it as:** "Datalog meets vector similarity search" - a foundation for approximate deductive reasoning at scale.

## 🌟 Key Features

### Deductive Reasoning
- **Horn clause rules** - Full support for `head :- body1, body2, ...` syntax
- **Forward chaining** - Iterative rule application with fixpoint detection
- **Transitive closure** - Multi-hop inference (arbitrary depth)
- **Semi-naive evaluation** - Optimized chaining that avoids redundant work
- **Negation support** - Classical negation (`~predicate`) and negation-as-failure (`not predicate`)
- **Stratification analysis** - Automatic detection of non-stratified programs with warnings

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
- **VSAR IDE** - Complete visual development environment (like DrRacket for VSAR)
- **VSARL language** - Declarative syntax for facts, queries, and rules
- **Interactive REPL** - Load files and query interactively
- **CLI interface** - Simple commands for ingestion, querying, and export
- **Full traceability** - Explanation DAG for debugging and transparency
- **Comprehensive testing** - 446 tests with 97.82% coverage

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install vsar

# Verify installation
vsar --help
```

### Development Install

```bash
# Install uv (fast Python package installer)
pip install uv

# Clone and install
git clone https://github.com/vasanthsarathy/vsar.git
cd vsar
uv sync

# For development, use uv run
uv run vsar --help
```

## 🚀 Quick Start

### Example 1: Basic Facts and Queries

Create `family.vsar`:

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

### Example 2: Reasoning with Rules (Phase 2)

Create `reasoning.vsar`:

```prolog
@model FHRR(dim=1024, seed=42);
@beam(width=50);
@novelty(threshold=0.95);

// Base facts: Parent relationships
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

### Example 3: Negation and Safety Analysis (Phase 3)

Create `safety.vsar`:

```prolog
@model FHRR(dim=1024, seed=42);
@beam(width=50);
@novelty(threshold=0.95);

// Positive facts: people and relationships
fact person(alice).
fact person(bob).
fact person(carol).
fact person(dave).

fact enemy(bob, dave).
fact criminal(dave).

// Negative facts: explicit statements about what is NOT true
fact ~criminal(alice).
fact ~criminal(bob).
fact ~enemy(alice, bob).

// Rule: A person is safe if they have no enemies
rule safe(X) :-
    person(X),
    not enemy(X, Y).

// Rule: A person is trustworthy if they are safe and not a criminal
rule trustworthy(X) :-
    safe(X),
    not criminal(X).

// Queries
query safe(X)?           // Who is safe?
query trustworthy(X)?    // Who is trustworthy?
query ~criminal(alice)?  // Check negative fact
```

Run it:

```bash
vsar run safety.vsar
```

Output:
```
Inserted 8 facts (4 positive, 4 negative)

Applied 2 rules in 1 iteration
Derived 3 new facts
Fixpoint reached: true

┌─────────────────┐
│ Query: safe(X)  │
├────────┬────────┤
│ Entity │ Score  │
├────────┼────────┤
│ alice  │ 0.9012 │
│ carol  │ 0.8954 │
└────────┴────────┘

┌──────────────────────┐
│ Query: trustworthy(X)│
├────────┬─────────────┤
│ Entity │ Score       │
├────────┼─────────────┤
│ alice  │ 0.8876      │
│ carol  │ 0.8754      │
└────────┴─────────────┘

┌──────────────────────────┐
│ Query: ~criminal(alice)  │
├────────┬─────────────────┤
│ Result │ Match           │
├────────┼─────────────────┤
│ ✓      │ Exact fact      │
└────────┴─────────────────┘
```

### Example 4: Interactive REPL

```bash
vsar repl
```

Example session:

```
VSAR Interactive REPL
Type 'help' for commands, 'exit' to quit

> load family.vsar
Loaded family.vsar
Inserted 3 facts

> query parent(alice, X)?
┌─────────────────────────┐
│ Query: parent(alice, X) │
├────────┬────────────────┤
│ Entity │ Score          │
├────────┼────────────────┤
│ bob    │ 0.9234         │
└────────┴────────────────┘

> stats
Knowledge Base Statistics
Total Facts: 3
Predicates: parent (3 facts)

> exit
Goodbye!
```

## 🎨 VSAR IDE - Interactive Development Environment

VSAR includes a complete visual IDE (similar to DrRacket) for an enhanced development experience.

### Launch the IDE

```bash
vsar-ide
```

### IDE Features

**Visual Interface:**
- **Split-pane layout** - Editor (left) + Console (right)
- **Syntax highlighting** - Real-time color-coding for VSARL
  - Keywords (`fact`, `rule`, `query`, `not`) in blue
  - Directives (`@model`, `@beam`) in purple
  - Negation operators (`~`, `not`) in crimson (bold)
  - Comments (`//`, `/* */`) in gray
  - Variables (uppercase) in orange
  - Predicates (lowercase) in black

**Development Workflow:**
- **File operations** - New (Ctrl+N), Open (Ctrl+O), Save (Ctrl+S), Save As (Ctrl+Shift+S)
- **Run programs** - Press F5 to execute entire programs
- **Interactive queries** - Ctrl+Q to run queries on-demand
- **KB statistics** - View facts and predicate breakdowns
- **Color-coded output** - Errors, successes, and warnings clearly marked

**Quick Start with IDE:**
1. Launch `vsar-ide`
2. Create a new file or open an example
3. Write your VSAR program
4. Press F5 to run
5. Use Ctrl+Q for interactive queries

**Perfect for:**
- Learning VSAR and experimenting
- Developing and testing programs
- Debugging with visual feedback
- Teaching and demonstrations

## 📚 Documentation

### Tutorials

- **[Getting Started](docs/getting-started.md)** - Your first VSAR program
- **[Tutorial: Family Tree Reasoning](examples/02_family_tree.vsar)** - Multi-hop inference
- **[Tutorial: Organizational Hierarchies](examples/04_organizational_hierarchy.vsar)** - Manager chains
- **[Tutorial: Knowledge Graphs](examples/05_knowledge_graph.vsar)** - Multi-relation reasoning

### User Guides

- **[VSARL Language Reference](docs/language.md)** - Complete syntax guide
- **[CLI Commands](docs/cli.md)** - Command-line interface
- **[Python API](docs/api.md)** - Programmatic usage
- **[Architecture Overview](docs/architecture.md)** - System design

### Reference

- **[API Reference](docs/api/)** - Complete API documentation
- **[Examples Directory](examples/)** - 7 example programs with explanations
- **[PROGRESS.md](PROGRESS.md)** - Current capabilities and limitations
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## 🎯 What Can VSAR Do?

### ✅ Currently Supported (Phase 0-3)

**Deductive Reasoning:**
- ✅ Ground facts insertion and querying
- ✅ Horn clause rules (`head :- body1, body2, ...`)
- ✅ Forward chaining with fixpoint detection
- ✅ Multi-hop inference (transitive closure)
- ✅ Recursive rules (arbitrary depth)
- ✅ Multiple interacting rules
- ✅ Semi-naive evaluation optimization
- ✅ Classical negation (`~predicate`) for explicit negative facts
- ✅ Negation-as-failure (`not predicate`) in rule bodies
- ✅ Stratification analysis with warnings for non-stratified programs

**Approximate Reasoning:**
- ✅ Similarity-based retrieval (fuzzy matching)
- ✅ Confidence scores for all results
- ✅ Graceful degradation under noise
- ✅ Top-k ranked results

**Performance:**
- ✅ Beam search joins (controls combinatorial explosion)
- ✅ Novelty detection (prevents duplicates)
- ✅ Vectorized operations (GPU-ready)
- ✅ Predicate partitioning

**Developer Experience:**
- ✅ Declarative VSARL language
- ✅ Interactive REPL
- ✅ CLI interface
- ✅ Full traceability and provenance
- ✅ HDF5 persistence

### ⏳ Limitations (Planned for Phase 4+)

- ⏳ **Single-variable queries only** - `parent(alice, ?)` works, `parent(?, ?)` doesn't yet
- ⏳ **No aggregation** - Cannot count, sum, max, etc.
- ⏳ **Forward chaining only** - No backward chaining or goal-directed search
- ⏳ **No magic sets** - Cannot optimize query-driven derivation
- ⏳ **Limited constraint solving** - No general constraint propagation

See [PROGRESS.md](PROGRESS.md) for detailed capability analysis and roadmap.

## 🏗️ Architecture

VSAR uses a layered architecture:

```
┌─────────────────────────────────────────┐
│         VSARL Language & CLI            │  (Phase 1)
│  Parser, AST, Engine, Trace, CLI        │
├─────────────────────────────────────────┤
│       Semantic Layer (Reasoning)        │  (Phase 2)
│  Substitution, Joins, Chaining, Rules   │
├─────────────────────────────────────────┤
│     Retrieval & Query Execution         │  (Phase 0)
│  Unbinding, Cleanup, Top-k Retrieval    │
├─────────────────────────────────────────┤
│        Knowledge Base (Storage)         │  (Phase 0)
│  Predicate Bundles, HDF5 Persistence    │
├─────────────────────────────────────────┤
│      Encoding (Role-Filler Binding)     │  (Phase 0)
│  Atom Encoding, Role Vector Management  │
├─────────────────────────────────────────┤
│    Symbol Registry (Typed Spaces)       │  (Phase 0)
│  E, R, A, C, T, S spaces + Basis Mgmt   │
├─────────────────────────────────────────┤
│    VSA Kernel (Hypervector Algebra)    │  (Phase 0)
│      FHRR, MAP Backends via VSAX        │
└─────────────────────────────────────────┘
```

**Key Principles:**
- **Approximate is explicit** - Every result has a similarity score
- **Modular semantics** - Clean separation of concerns
- **Bounded inference** - Beam widths, hop limits, novelty thresholds
- **Typed symbol spaces** - Entities (E), Relations (R), Attributes (A), etc.

## 📖 VSARL Language

### Facts

```prolog
// Positive facts
fact parent(alice, bob).
fact parent(bob, carol).
fact lives_in(alice, boston).
fact transfer(alice, bob, money).   // Ternary fact
fact person(alice).                  // Unary fact

// Negative facts (Phase 3)
fact ~enemy(alice, bob).             // Alice is NOT an enemy of Bob
fact ~criminal(alice).               // Alice is NOT a criminal
```

### Rules (Phase 2-3)

```prolog
// Grandparent: X is grandparent of Z if X is parent of Y and Y is parent of Z
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

// Ancestor: Base case
rule ancestor(X, Y) :- parent(X, Y).

// Ancestor: Recursive case (transitive closure)
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

// Sibling: Share same parent
rule sibling(X, Y) :- parent(Z, X), parent(Z, Y).

// Negation-as-failure (Phase 3): Safe person has no enemies
rule safe(X) :- person(X), not enemy(X, Y).

// Multiple NAF literals: Trustworthy person is safe and not a criminal
rule trustworthy(X) :- safe(X), not criminal(X).
```

### Queries

```prolog
query parent(alice, X)?         // Find children of alice
query parent(X, carol)?         // Find parents of carol
query grandparent(alice, X)?    // Find grandchildren of alice (via rules)
query ancestor(alice, X)?       // Find all descendants (transitive)
query safe(X)?                  // Find safe people (via NAF rules)
query ~criminal(alice)?         // Check negative fact
```

### Directives

```prolog
// Model configuration
@model FHRR(dim=1024, seed=42);    // FHRR backend, 1024 dimensions
@model MAP(dim=512, seed=100);     // MAP backend (alternative)

// Retrieval parameters
@threshold 0.22;                   // Similarity threshold
@beam(width=50);                   // Beam width for joins
@novelty(threshold=0.95);          // Novelty detection threshold
```

### Comments

```prolog
// Single-line comment

/* Multi-line
   comment */
```

## 🔧 CLI Reference

### Run Programs

```bash
# Run a VSAR program
vsar run program.vsar

# Limit results per query
vsar run program.vsar --k 10

# JSON output (for scripting)
vsar run program.vsar --json

# Show trace DAG
vsar run program.vsar --trace
```

### Ingest Facts

```bash
# From CSV (predicate in first column)
vsar ingest facts.csv --kb family.h5

# From CSV (specify predicate)
vsar ingest parents.csv --predicate parent --kb family.h5

# From JSONL
vsar ingest facts.jsonl --kb family.h5
```

### Export & Inspect

```bash
# Export KB to JSON
vsar export family.h5 --format json --output facts.json

# Export to JSONL
vsar export family.h5 --format jsonl --output facts.jsonl

# Inspect KB statistics
vsar inspect family.h5
```

### Interactive REPL

```bash
# Start interactive session
vsar repl

# Available commands:
# - load <file>      Load a VSAR program
# - query <query>    Execute a query
# - stats            Show KB statistics
# - help             Show help
# - exit             Exit REPL
```

## 🐍 Python API

### High-Level API (Recommended)

```python
from vsar.language.ast import Directive, Fact, Query, Rule, Atom
from vsar.semantics.engine import VSAREngine

# Configure engine
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 1024, "seed": 42}),
    Directive(name="beam", params={"width": 50}),
    Directive(name="novelty", params={"threshold": 0.95}),
]
engine = VSAREngine(directives)

# Insert facts
engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

# Define rules
rules = [
    Rule(
        head=Atom(predicate="grandparent", args=["X", "Z"]),
        body=[
            Atom(predicate="parent", args=["X", "Y"]),
            Atom(predicate="parent", args=["Y", "Z"]),
        ],
    )
]

# Query with automatic rule application
query = Query(predicate="grandparent", args=["alice", None])
result = engine.query(query, rules=rules, k=10)

for entity, score in result.results:
    print(f"{entity}: {score:.4f}")

# Inspect trace
trace = engine.trace.get_dag()
for event in trace:
    print(f"{event.type}: {event.payload}")

# Get KB statistics
stats = engine.stats()
print(f"Total facts: {stats['total_facts']}")

# Save/load KB
engine.save_kb("family.h5")
engine.load_kb("family.h5")
```

### Forward Chaining

```python
from vsar.semantics.chaining import apply_rules

# Apply rules with forward chaining
result = apply_rules(
    engine,
    rules,
    max_iterations=100,
    k=10,
    semi_naive=True  # Use semi-naive evaluation (faster)
)

print(f"Iterations: {result.iterations}")
print(f"Total derived: {result.total_derived}")
print(f"Fixpoint reached: {result.fixpoint_reached}")
```

### Loading from Files

```python
from vsar.language.loader import ProgramLoader

# Load VSAR program
loader = ProgramLoader()
program = loader.load_file("examples/02_family_tree.vsar")

# Create engine from program directives
engine = VSAREngine(program.directives)

# Insert all facts
for fact in program.facts:
    engine.insert_fact(fact)

# Execute all queries with rules
for query in program.queries:
    result = engine.query(query, rules=program.rules, k=10)
    print(f"Query: {query.predicate}({', '.join(str(a) for a in query.args)})")
    print(f"Results: {result.results}")
```

## 📊 Performance

**Approximate query performance** (Phase 2, with rules):

| Facts | Query Time | Chaining Time (10 rules) |
|-------|------------|--------------------------|
| 10^3  | <50ms      | <200ms                   |
| 10^4  | <100ms     | <500ms                   |
| 10^5  | <300ms     | <2s                      |
| 10^6  | <800ms     | <10s                     |

*Measured on AMD EPYC 7742 CPU with dim=1024, beam=50*

**Memory usage:**
- Base: ~50MB (dim=1024)
- Per 1000 facts: ~5MB
- Scales linearly with fact count and dimensionality

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vsar --cov-report=html

# Run specific test suites
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/integration/test_e2e_phase2.py  # End-to-end tests
```

**Test Statistics:**
- **446 tests** (all passing, 4 skipped)
- **97.82% coverage**
- Unit tests: 392
- Integration tests: 36
- Negation tests: 54
- End-to-end tests: 5

## 🗺️ Project Status & Roadmap

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

### ✅ Phase 3: Negation Support (Complete)
- Classical negation (`~predicate`) for explicit negative facts
- Negation-as-failure (`not predicate`) in rule bodies
- Stratification analysis and cycle detection
- Paraconsistent logic (allows contradictions)
- Warnings for non-stratified programs

### 🔜 Phase 4: Advanced Features (Planned)
- Multi-variable queries (`parent(?, ?)?`)
- Aggregation (count, sum, max)
- Backward chaining
- Magic sets optimization

### 🔜 Phase 5: Scale & Performance (Planned)
- Incremental maintenance
- Query planning and optimization
- Parallel execution
- GPU acceleration

See [PROGRESS.md](PROGRESS.md) for detailed status and comparison to other reasoners.

## 💡 Use Cases

**Best suited for:**
1. Knowledge graph reasoning with noise tolerance
2. Transitive closure queries (org hierarchies, supply chains)
3. Multi-hop reasoning (family trees, social networks)
4. Explainable AI (need provenance and similarity scores)
5. Large-scale approximate reasoning (vectorized operations)

**Not yet suitable for:**
1. Planning problems (need backward chaining)
2. Ontology reasoning (need DL features)
3. Complex answer set programming (need choice rules, optimization)
4. Real-time reasoning (optimization needed)

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Development setup:**

```bash
# Clone repo
git clone https://github.com/vasanthsarathy/vsar.git
cd vsar

# Install with dev dependencies
uv sync --all-groups

# Run tests
uv run pytest

# Format code
uv run black .
uv run ruff check . --fix

# Type check
uv run mypy src/vsar
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 📚 Citation

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

## 🙏 Acknowledgments

- Built on [VSAX](https://vsarathy.com/vsax/) for VSA operations
- Inspired by Datalog, Prolog, and logic programming systems
- Uses [Lark](https://github.com/lark-parser/lark) for parsing
- CLI powered by [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)
- Testing with [pytest](https://pytest.org/)

## 📞 Support

- **Documentation:** [docs/](docs/)
- **Examples:** [examples/](examples/)
- **Issues:** [GitHub Issues](https://github.com/vasanthsarathy/vsar/issues)
- **Discussions:** [GitHub Discussions](https://github.com/vasanthsarathy/vsar/discussions)

---

**Made with ❤️ for approximate reasoning at scale.**
