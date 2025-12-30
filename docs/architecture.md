# VSAR Architecture

This document describes the complete architecture of VSAR (VSAX Reasoner), a VSA-grounded reasoning system built on hypervector algebra.

## Overview

VSAR implements approximate unification and retrieval using Vector Symbolic Architectures (VSA). The system is organized into 9 layers across 2 phases:

```
┌─────────────────────────────────────────────┐
│          CLI Layer (Phase 1)                 │
│  (Typer commands, Rich formatting)           │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│          Semantics Layer (Phase 1)           │
│  (VSAREngine, orchestration)                 │
└─────────────────────────────────────────────┘
                    ▲
      ┌─────────────┴─────────────┐
      ▼                           ▼
┌─────────────────┐    ┌─────────────────────┐
│  Language       │    │  Trace Layer        │
│  (Phase 1)      │    │  (Phase 1)          │
│  Parser, AST,   │    │  Explanation DAG    │
│  Loaders        │    │                     │
└─────────────────┘    └─────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│          Retrieval Layer (Phase 0)           │
│  (unbind, cleanup, top-k query)              │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│          KB Storage Layer (Phase 0)          │
│  (predicate-partitioned bundles, HDF5)       │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│          Encoding Layer (Phase 0)            │
│  (role-filler binding, VSA encoder)          │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│          Symbol Layer (Phase 0)              │
│  (typed spaces, basis generation, registry)  │
└─────────────────────────────────────────────┘
                    ▲
┌─────────────────────────────────────────────┐
│          Kernel Layer (Phase 0)              │
│  (FHRR via VSAX, bind/unbind/bundle)         │
└─────────────────────────────────────────────┘
```

## Phase 0 Layers (Foundation)

### 1. Kernel Layer (`vsar.kernel`)

Provides low-level VSA operations via the VSAX library.

**Key Components:**
- `KernelBackend` (abstract): Interface for VSA operations
- `FHRRBackend`: FHRR (Fourier Holographic Reduced Representations) implementation
  - Uses complex-valued vectors
  - Bind: Circular convolution via FFT
  - Unbind: Complex conjugate + normalization
  - Bundle: Element-wise sum + normalization
  - Similarity: Cosine similarity
- `MAPBackend`: MAP (Multiply-Add-Permute) implementation

**Design Decisions:**
- Strategy pattern enables polymorphic backend usage
- All operations return normalized vectors
- Wraps VSAX models for GPU acceleration
- Deterministic generation with JAX PRNGKey

**Code Location:** `src/vsar/kernel/`

### 2. Symbol Layer (`vsar.symbols`)

Manages typed symbol spaces and basis vector generation.

**Key Components:**
- `SymbolSpace` enum: Six typed spaces (E, R, A, C, T, S)
  - **E** (Entities): alice, bob, carol
  - **R** (Relations): parent, sibling
  - **A** (Attributes): color, age
  - **C** (Contexts): historical, hypothetical
  - **T** (Time): timestamps, intervals
  - **S** (Structural): list constructors, tuples

- `SymbolRegistry`: Central registry for all symbols
  - Lazy generation: symbols created on first access
  - Cleanup: Reverse lookup via similarity search
  - HDF5 persistence: Save/load basis vectors

- `generate_basis()`: Deterministic hypervector generation
  - Seed derivation: `hash(space) + hash(name) + seed`
  - Ensures same symbol → same vector across sessions

**Design Decisions:**
- Typed spaces prevent collisions (e.g., entity "parent" vs. relation "parent")
- Deterministic generation enables reproducibility
- Registry pattern centralizes symbol management

**Code Location:** `src/vsar/symbols/`

### 3. Encoding Layer (`vsar.encoding`)

Encodes logical atoms into hypervectors using role-filler binding.

**Key Components:**
- `RoleVectorManager`: Manages role vectors (ρ1, ρ2, ...)
  - Orthogonal roles for different argument positions
  - Deterministic with seed offset (base_seed + 10000 + position)

- `VSAEncoder`: Role-filler binding implementation
  - Formula: `enc(p(t1,...,tk)) = hv(p) ⊗ ((hv(ρ1) ⊗ hv(t1)) ⊕ ... ⊕ (hv(ρk) ⊗ hv(tk)))`
  - Query encoding: Supports None for variables
  - Example: `parent(alice, X)` → `hv(parent) ⊗ (hv(ρ1) ⊗ hv(alice))`

**Design Decisions:**
- Role vectors distinguish argument positions
- Query patterns use None for unbound variables
- Automatic symbol registration in appropriate spaces

**Code Location:** `src/vsar/encoding/`

### 4. KB Storage Layer (`vsar.kb`)

Stores ground atoms as bundled hypervectors, partitioned by predicate.

**Key Components:**
- `KnowledgeBase`: Predicate-partitioned storage
  - Structure: `dict[predicate_name, bundled_hypervector]`
  - Incremental bundling: `new_bundle = old_bundle ⊕ new_atom`
  - Fact lists: `dict[predicate_name, list[fact_tuples]]`

- `save_kb()` / `load_kb()`: HDF5 persistence
  - Bundles stored as datasets
  - Facts stored as JSON attributes
  - Preserves insertion order

**Design Decisions:**
- Predicate partitioning reduces noise during retrieval
- HDF5 format enables efficient large-scale storage
- Maintains both vectors (for retrieval) and facts (for reference)

**Code Location:** `src/vsar/kb/`

### 5. Retrieval Layer (`vsar.retrieval`)

Implements top-k retrieval via unbinding and cleanup.

**Key Components:**
- `unbind_query_from_bundle()`: Extract relevant facts
  - `bundle ⊗^(-1) query → role-filler pairs`

- `unbind_role()`: Isolate entity from role-filler binding
  - `(ρ ⊗ entity) ⊗^(-1) ρ → entity`

- `cleanup()`: Nearest neighbor search
  - Compute similarity against all basis vectors in space
  - Return top-k matches sorted by score

- `Retriever`: High-level query interface
  - `retrieve(predicate, var_position, bound_args, k)` → top-k results
  - Orchestrates: encode query → get bundle → unbind → cleanup

**Retrieval Pipeline:**

```
Query: parent(alice, X)
    ↓
1. Encode query: hv(parent) ⊗ (hv(ρ1) ⊗ hv(alice))
    ↓
2. Get KB bundle: hv(parent(alice,bob)) ⊕ hv(parent(bob,carol)) ⊕ ...
    ↓
3. Unbind query from bundle: bundle ⊗^(-1) query
    → Contains: hv(ρ2 ⊗ bob) + noise
    ↓
4. Unbind role ρ2: result ⊗^(-1) ρ2
    → ~hv(bob) + noise
    ↓
5. Cleanup: find top-k nearest in ENTITIES space
    → [("bob", 0.85), ("carol", 0.42), ...]
```

**Code Location:** `src/vsar/retrieval/`

## Phase 1 Layers (Language & CLI)

### 6. Language Layer (`vsar.language`)

Parses VSARL programs and loads facts from multiple formats.

**Key Components:**
- **Grammar** (`grammar.lark`): EBNF-style Lark grammar
  - Directives: `@model FHRR(dim=8192, seed=42);`
  - Facts: `fact parent(alice, bob).`
  - Queries: `query parent(alice, X)?`
  - Comments: `//` and `/* */`

- **AST** (`ast.py`): Pydantic-based abstract syntax tree
  - `Directive(name, params)`: Configuration directives
  - `Fact(predicate, args)`: Ground atoms
  - `Query(predicate, args)`: Queries with variables (None = variable)
  - `Program(directives, facts, queries)`: Complete program

- **Parser** (`parser.py`): Lark wrapper with transformer
  - `parse(text) → Program`: Parse VSARL string
  - `parse_file(path) → Program`: Parse .vsar file

- **Loaders** (`loader.py`): Multi-format fact ingestion
  - `load_csv(path, predicate)`: CSV format
  - `load_jsonl(path)`: JSONL format
  - `load_vsar(path)`: Native .vsar format
  - `load_facts(path, format="auto")`: Unified loader with auto-detection

**VSARL Language Specification:**

```ebnf
program     := statement*
statement   := directive | fact | query

directive   := "@" IDENTIFIER (IDENTIFIER "(" params ")" | "(" params ")") ";"
params      := param ("," param)*
param       := IDENTIFIER "=" value
value       := NUMBER | STRING | IDENTIFIER | TRUE | FALSE

fact        := "fact" atom "."
query       := "query" atom "?"
atom        := predicate "(" args? ")"
predicate   := LOWER_NAME
args        := arg ("," arg)*
arg         := constant | variable
constant    := LOWER_NAME
variable    := UPPER_NAME
```

**Design Decisions:**
- Lark provides declarative grammar with good error messages
- Pydantic AST enables type checking and validation
- Auto-format detection simplifies user experience
- Variables represented as None in Query.args for simplicity

**Code Location:** `src/vsar/language/`

### 7. Semantics Layer (`vsar.semantics`)

Orchestrates all layers to execute VSAR programs.

**Key Components:**
- `VSAREngine`: Main execution engine
  - Initialization: Parses directives, creates backend/registry/encoder/kb/retriever
  - `insert_fact(fact)`: Encodes and inserts fact into KB
  - `query(query, k)`: Executes query with tracing, returns top-k results
  - `stats()`: Returns KB statistics (total facts, predicates)
  - `export_kb(format)`: Exports KB to JSON/JSONL
  - `save_kb(path)` / `load_kb(path)`: HDF5 persistence

- `QueryResult`: Result container
  - `query`: Original query
  - `results`: List of (entity, score) tuples
  - `trace_id`: ID for trace DAG lookup

**Directive Processing:**
- `@model FHRR(dim=8192, seed=42)` → Creates FHRRBackend
- `@threshold(value=0.22)` → Sets retrieval threshold
- `@beam(width=50)` → Sets beam width (Phase 2)

**Design Decisions:**
- Engine encapsulates all Phase 0 components for ease of use
- Trace collection integrated into query execution
- Supports both FHRR and MAP backends
- Statistics API for monitoring KB growth

**Code Location:** `src/vsar/semantics/`

### 8. Trace Layer (`vsar.trace`)

Builds explanation DAG for transparency and debugging.

**Key Components:**
- `TraceEvent`: Single event in execution trace
  - `id`: Unique event identifier (UUID)
  - `type`: Event type (query, unbind, cleanup, retrieval)
  - `payload`: Event-specific data (dict)
  - `parent_ids`: List of parent event IDs
  - `timestamp`: Event creation time

- `TraceCollector`: DAG builder
  - `record(type, payload, parent_ids) → event_id`: Record new event
  - `get_dag() → list[TraceEvent]`: Get all events chronologically
  - `get_subgraph(event_id) → list[TraceEvent]`: Get event + ancestors
  - `to_dict() / from_dict()`: Serialization support

**Event Types:**
- **query**: Query encoding operation
  - Payload: `{predicate, args, variables, bound_args}`
- **retrieval**: Final retrieval results
  - Payload: `{predicate, var_position, k, num_results, results}`

**Design Decisions:**
- DAG structure captures causal relationships between operations
- UUID-based IDs for stable references
- Serializable for export/analysis
- Subgraph extraction for focused debugging

**Code Location:** `src/vsar/trace/`

### 9. CLI Layer (`vsar.cli`)

Typer-based command-line interface with Rich formatting.

**Key Components:**
- **Main App** (`main.py`): Four commands
  - `vsar run program.vsar`: Execute VSAR program
  - `vsar ingest facts.csv --kb kb.h5`: Ingest facts
  - `vsar export kb.h5 --format json`: Export KB
  - `vsar inspect kb.h5`: Show statistics

- **Formatters** (`formatters.py`): Output formatting
  - `format_results_table()`: Rich table for query results
  - `format_results_json()`: JSON for scripting
  - `format_trace_dag()`: Trace DAG visualization
  - `format_stats()`: KB statistics tables

**CLI Options:**
- `--json`: JSON output instead of tables
- `--trace`: Show trace DAG
- `--k N`: Limit results per query
- `--predicate P`: Specify predicate for CSV
- `--format F`: Specify format (auto/csv/jsonl/vsar)

**Design Decisions:**
- Typer provides automatic help generation and validation
- Rich enables beautiful terminal output
- JSON mode for integration with other tools
- Entry point in pyproject.toml: `vsar = "vsar.cli.main:app"`

**Code Location:** `src/vsar/cli/`

## Data Flow Examples

### High-Level API (Recommended)

Complete example using VSAREngine:

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.language.loader import load_facts
from vsar.semantics.engine import VSAREngine

# 1. Create engine from directives
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
]
engine = VSAREngine(directives)

# 2. Load and insert facts
facts = load_facts("family.csv")  # Auto-detects CSV format
for fact in facts:
    engine.insert_fact(fact)
# Internally: encoder.encode_atom() → kb.insert()

# 3. Execute query
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=5)
# Internally: retriever.retrieve() + trace.record()

# 4. Process results
for entity, score in result.results:
    print(f"{entity}: {score:.4f}")

# 5. Inspect trace
for event in engine.trace.get_dag():
    print(f"{event.type}: {event.payload}")

# 6. Save KB
engine.save_kb("family.h5")
```

### CLI Workflow

```bash
# Create VSAR program
cat > family.vsar <<EOF
@model FHRR(dim=8192, seed=42);
@threshold(value=0.22);

fact parent(alice, bob).
fact parent(bob, carol).

query parent(alice, X)?
EOF

# Execute program
vsar run family.vsar

# Output:
# Inserted 2 facts
#
# ┌─────────────────────────┐
# │ Query: parent(alice, X) │
# ├────────┬────────────────┤
# │ Entity │ Score          │
# ├────────┼────────────────┤
# │ bob    │ 0.9234         │
# └────────┴────────────────┘

# Export to JSON
vsar export family.h5 --format json --output facts.json

# Inspect statistics
vsar inspect family.h5
```

### Low-Level API (Phase 0)

Direct usage of Phase 0 layers:

```python
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.encoding.roles import RoleVectorManager
from vsar.kb.store import KnowledgeBase
from vsar.retrieval.query import Retriever

# 1. Setup system
backend = FHRRBackend(dim=512, seed=42)
registry = SymbolRegistry(backend, seed=42)
encoder = VSAEncoder(backend, registry, seed=42)
kb = KnowledgeBase(backend)
role_manager = RoleVectorManager(backend, seed=42)
retriever = Retriever(backend, registry, kb, encoder, role_manager)

# 2. Insert facts
atom_vec = encoder.encode_atom("parent", ["alice", "bob"])
kb.insert("parent", atom_vec, ("alice", "bob"))

atom_vec2 = encoder.encode_atom("parent", ["bob", "carol"])
kb.insert("parent", atom_vec2, ("bob", "carol"))

# 3. Query: parent(alice, X)
results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)
# [("bob", 0.85), ...]
```

## File Format Specifications

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

**Loader behavior:**
- Strips whitespace from all fields
- Skips empty rows
- Validates minimum 2 columns when predicate is in first column

### JSONL Format

One fact per line (JSON Lines):
```jsonl
{"predicate": "parent", "args": ["alice", "bob"]}
{"predicate": "parent", "args": ["bob", "carol"]}
{"predicate": "lives_in", "args": ["alice", "boston"]}
```

**Validation:**
- Each line must be valid JSON object
- Required fields: `predicate` (string), `args` (array)
- All args converted to strings

### VSAR Format

Native `.vsar` program files:
```prolog
@model FHRR(dim=8192, seed=42);
@threshold(value=0.22);

fact parent(alice, bob).
fact parent(bob, carol).

query parent(alice, X)?
query parent(X, carol)?
```

## Determinism and Reproducibility

VSAR ensures deterministic behavior across all layers:

1. **Fixed seeds** → identical random vectors (Kernel)
2. **Hash-based** seed derivation → same symbols always get same vectors (Symbols)
3. **Deterministic VSAX** operations (no stochastic elements) (Kernel)
4. **Order-independent** bundling (commutative sum) (KB)
5. **Deterministic parsing** (Lark grammar) (Language)
6. **Deterministic UUIDs** (based on event content) - *Future enhancement*

Regression tests verify:
- Same seed produces same encodings
- Same seed produces same retrieval results
- Repeated runs produce identical outputs
- KB persistence preserves exact state

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Parse program | O(n) | n = program size |
| Load CSV | O(m) | m = file size |
| Encode atom | O(k·d) | k = arity, d = dimension |
| Insert fact | O(d) | Bundle operation |
| Query (Phase 1) | O(d + s·d + k·log k) | s = symbols, k = top-k |
| Cleanup | O(s·d) | s = symbols in space |

### Space Complexity

| Component | Size | Notes |
|-----------|------|-------|
| Hypervector | d floats | Typically d = 512-8192 |
| KB bundle | d floats per predicate | One bundle per predicate |
| Symbol registry | s·d floats | s = total unique symbols |
| AST | O(n) | n = number of statements |
| Trace DAG | O(e) | e = number of events |

### Approximate Retrieval Performance

**Phase 1 Query Performance:**
- 10^3 facts: <100ms per query
- 10^4 facts: <200ms per query
- 10^5 facts: <500ms per query
- 10^6 facts: <1s per query

*Measured on AMD EPYC 7742 CPU with dim=8192*

**Retrieval Quality:**
- Bind/unbind fidelity: ~50% similarity (acceptable for VSA)
- Cleanup threshold: >0.22 similarity (configurable via @threshold)
- Noise tolerance: Bundling 10-100 facts per predicate maintains good retrieval

## Extension Points

### Adding New File Formats

To add XML fact loading:

```python
def load_xml(path: Path) -> list[Fact]:
    tree = ET.parse(path)
    facts = []
    for fact_elem in tree.findall(".//fact"):
        predicate = fact_elem.get("predicate")
        args = [arg.text for arg in fact_elem.findall("arg")]
        facts.append(Fact(predicate=predicate, args=args))
    return facts

# Register in load_facts()
```

### Adding New Directives

To add `@cache` directive:

```python
# In VSAREngine._parse_config()
elif directive.name == "cache":
    config["cache_size"] = directive.params.get("size", 1000)
    config["cache_policy"] = directive.params.get("policy", "LRU")
```

### Adding New Trace Event Types

```python
# In retrieval layer
trace.record(
    "unbind",
    {
        "operation": "query_from_bundle",
        "bundle_size": kb.count(predicate),
        "similarity": sim_score,
    },
    parent_ids=[query_event_id],
)
```

### Future Backend Support

To add Clifford algebra backend:

1. Implement `CliffordBackend(KernelBackend)`
2. Define geometric product for bind
3. Implement reverse/conjugate for unbind
4. Update `VSAREngine._parse_config()` to recognize "Clifford" type

## Testing Strategy

VSAR has comprehensive test coverage across both phases:

### Unit Tests (261 tests)

**Phase 0 (156 tests):**
- Kernel operations (bind, unbind, bundle)
- Symbol registration and cleanup
- Encoding (atoms, queries, roles)
- KB storage and persistence
- Retrieval primitives

**Phase 1 (105 tests):**
- Language parsing (facts, queries, directives, comments)
- Loaders (CSV, JSONL, VSAR, auto-detection)
- Trace events and DAG construction
- Engine initialization and execution
- CLI formatters

### Integration Tests (20 tests)

**Phase 0 (9 tests):**
- End-to-end VSA flow
- Persistence workflows (save/load)
- Determinism regression tests

**Phase 1 (11 tests):**
- Complete VSAR program execution
- CSV/JSONL ingestion and querying
- KB persistence with engine
- Export functionality
- Trace DAG construction
- Large-scale ingestion (1000+ facts)

### Coverage Targets
- **Minimum**: 90% coverage (enforced)
- **Current**: 98.5% coverage
- **Excluded**: CLI main.py (argument parsing), type definitions, version file

### Test Command
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=vsar --cov-report=html

# Run specific suites
uv run pytest tests/unit/language/     # Language layer
uv run pytest tests/integration/       # Integration tests
```

## Design Principles

### Layered Architecture
- Each layer has single responsibility
- Dependencies flow downward (CLI → Semantics → Phase 0)
- Phase 0 layers are immutable (no modifications in Phase 1)

### Separation of Concerns
- Language: Parsing and representation
- Semantics: Orchestration and business logic
- Trace: Observability and debugging
- CLI: User interaction

### Type Safety
- Pydantic for AST validation
- MyPy type checking throughout codebase
- Typed symbol spaces prevent errors

### Testability
- Unit tests for each layer
- Integration tests for workflows
- High coverage (98.5%)

### Extensibility
- Strategy pattern for backends
- Plugin architecture for loaders
- Directive system for configuration

## References

- [VSAX Library](https://vsarathy.com/vsax/) - GPU-accelerated VSA operations
- [FHRR Paper](https://www.researchgate.net/publication/2884506_Holographic_Reduced_Representation) - Plate (2003)
- [VSA Survey](https://arxiv.org/abs/2001.11797) - Kleyko et al. (2021)
- [Hyperdimensional Computing](https://redwood.berkeley.edu/wp-content/uploads/2020/08/Kanerva2009.pdf) - Kanerva (2009)
- [Lark Parser](https://github.com/lark-parser/lark) - Declarative parsing library
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting
