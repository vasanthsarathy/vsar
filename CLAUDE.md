# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VSAR (VSAX Reasoner) is a VSA-grounded reasoning tool that provides a unified inference substrate using hypervector algebra for deductive reasoning, approximate joins, and explainable results. It addresses classical Datalog/Prolog/ASP limitations through approximate reasoning with massive speedups via vectorized operations.

**Key Features:**
- Deductive reasoning over facts and Horn-style rules (Datalog-like)
- Approximate unification via VSA binding/unbinding and similarity search
- Explainable results with similarity scores, retrieved candidates, and rule firings
- Supports both classical VSA and Clifford algebra modes

## Architecture

VSAR is designed as a **three-layer system**:

1. **Formal core** (language + semantics)
2. **Python library** (`vsar` package)
3. **Interactive web playground** (React + TypeScript frontend)

### High-level Components

1. **Language front-end**: Lexer/parser → AST → Type checker → Compiler → execution plan
2. **VSA inference kernel**: Symbol bases, encoders, store (bundled KB + indexes), retrieval, cleanup/factorization
3. **Execution engine**: Plan runner (vectorized), bounded search (beam), caching
4. **Trace & explanation layer**: Proof graph with scores and thresholds
5. **UI**: Editor + runner + results explorer + trace viewer

### Kernel Operations (IR)

The compiler emits an IR graph with these operations:
- `SYM(space, name)` → symbol handle
- `ENC_ATOM(pred, args[])` → atom vector
- `STORE(pred, vec)` → insert into KB bundle/index
- `RETRIEVE(space, vec, k)` → top-k nearest symbols
- `UNBIND(role_or_factor, vec)` → factor isolation
- `JOIN(var, cand_sets, beam)` → approximate join
- `FIRE(rule_id, subst)` → derive head atoms
- `NOVELTY(pred, vec, θ_novel)` → membership test
- `TRACE(event, payload)` → trace emission

## Tech Stack

### Backend (Python 3.11+)
- **VSAX**: Reasoning substrate (hypervectors, models, encoders, memory, factorization)
- **JAX**: GPU/CPU vectorization
- **FastAPI**: Server mode and UI API
- **SQLite / DuckDB**: Metadata and persistence (optional)
- **FAISS / hnswlib**: ANN indexing for large-scale nearest neighbors (optional)
- **Typer**: CLI interface

### Frontend (Web UI)
- **React + TypeScript**
- **Monaco Editor**: VSARL language editor
- **Tailwind**: Styling
- **IBM Plex Mono**: Typography
- **Black/white theme** with semantic colors only for alerts

### Module Structure

```
vsar/
  language/        # grammar, AST, parser
  semantics/       # formal semantics, execution engine
  kernel/          # VSA / Clifford backends (VSAX-based)
  kb/              # KB storage, indexing, novelty detection
  trace/           # trace graph structures
  cli/             # CLI entry points
  utils/
```

## Development Commands

### Python Setup
```bash
# Install dependencies (using uv/pip)
pip install vsar

# Development install
pip install -e .
```

### CLI Commands
```bash
# Run a VSAR program
vsar run program.vsar

# Interactive REPL
vsar repl

# Ingest facts from JSONL
vsar ingest facts.jsonl --predicate parent

# Export KB
vsar export --format json

# Inspect KB
vsar inspect kb

# Start server mode
vsar serve --port 8080
```

### Testing
```bash
# Run full test suite
pytest

# Run with coverage (target ≥90%)
pytest --cov=vsar --cov-report=html

# Run specific test types
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/semantic/      # Semantic tests
pytest tests/regression/    # Regression tests
```

### Code Quality
```bash
# Format Python code
black .

# Lint Python code
ruff check .

# Type check
mypy vsar/

# Run pre-commit hooks
pre-commit run --all-files
```

### Frontend Development
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Lint TypeScript
npm run lint

# Format code
npm run format
```

## VSARL Language (VSA Reasoning Language)

### Syntax Examples

**Facts:**
```
fact parent(alice, bob).
fact lives_in(bob, boston).
```

**Rules (Horn):**
```
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

**Queries:**
```
query grandparent(alice, Z)?
```

**Directives:**
```
@model FHRR(dim=8192, seed=1);
@threshold 0.22;
@beam 50;
@max_hops 3;
@trace full;
```

**Negation:**
```
fact !enemy(alice, bob).              # Classical negation
rule safe(X) :- person(X), not enemy(X, _).  # Negation-as-failure
```

### Identifiers
- **Predicates**: lower_snake (e.g., `parent`, `lives_in`)
- **Constants**: lowercase (e.g., `bob`, `boston`)
- **Variables**: UpperCamel (e.g., `X`, `Person`)

## Design Principles

1. **Approximation is explicit**: Every inference emits `(answer, score, trace)`
2. **Semantics are modular**: Front-end language compiles to kernel operations
3. **Typed symbol spaces**: Entities (E), Relations (R), Attributes (A), Contexts (C), Time (T), Structural (S)
4. **Bounded inference**: Beam widths, hop limits, novelty thresholds prevent blowups
5. **Two-tier execution**:
   - Fast path: vector retrieval (unbinding + similarity)
   - Slow path: factorization / resonator decoding (optional, bounded)

## VSA vs Clifford Mode

**Use VSA mode when:**
- Maximum throughput and simplicity needed
- Mainly set-like relational facts with shallow arity

**Use Clifford mode when:**
- Order-sensitive structures (sequences, nested terms, paths)
- More stable factorization for multi-hop explanations
- Structural invariants (symmetry/orientation tests) in traces
- Typed decomposition via grade projections

**Key difference:**
- VSA bundling: `a ⊕ b = b ⊕ a` (commutative, order erased)
- Clifford geometric product: `a ⋆ b ≠ b ⋆ a` (captures orientation/order)

## Python API Example

```python
from vsar import Program, Engine

# Load and run program
prog = Program.from_file("example.vsar")
engine = Engine(model="FHRR", dim=8192)

result = engine.run(prog)
print(result.answers)
print(result.trace)
```

## Software Engineering Practices

### Version Control
- Main branches: `main` (stable), `dev` (active development)
- Feature branches: `feature/*`, `fix/*`
- All changes via pull requests
- Small, atomic commits with descriptive messages

### Code Quality Requirements
- Formatting: `black` (Python), `prettier` (TS/JS)
- Linting: `ruff` (Python), `eslint` (frontend)
- Type checking: `mypy` (Python), TypeScript strict mode
- Pre-commit hooks required (formatting, lint, type checks, security)

### Testing Requirements
- Target ≥90% line coverage for core modules
- Fixed seeds for reproducibility in regression tests
- CI enforces coverage thresholds and blocks merges on failure

### Versioning
- Semantic versioning: `MAJOR.MINOR.PATCH`
- Language versioning tracked independently (e.g., `@lang 0.1`)

## UI/UX Design

### Visual Language
- **Monochrome**: black/white only
- **Accent colors** reserved for: errors (red), warnings (amber), success (green), active selection (blue)
- **Typography**: IBM Plex Mono everywhere
- **Shapes**: sharp corners, no rounding
- **Spacing**: tight, minimal padding (2–6px)
- **Borders**: 1px high-contrast lines

### Layout (Three-panel IDE)
1. **Left sidebar**: KB browser, symbol spaces, run history
2. **Center**: Monaco editor + toolbar + console output
3. **Right**: Query results table + trace viewer + fact inspector

## Implementation Phases

### Phase 0 — Foundation (Weeks 1–2)
- VSAX model selection + symbol bases + persistence
- Atom encoder + KB storage (predicate bundles)
- Retrieval query primitive with top-k results

### Phase 1 — Ground KB + conjunctive queries (Weeks 3–6)
- Facts ingestion (CSV/JSONL/VSAR source)
- Predicate-local bundles and indexes
- Query execution: unbind → retrieve → score
- Trace output

### Phase 2 — Horn rules + bounded chaining (Weeks 7–12)
- Rule compilation to retrieval plans
- Beam search joins, novelty detection, semi-naive caching
- Derived fact store + provenance traces

### Phase 3+ — Post-MVP
- Negation + defaults
- Argumentation + epistemic contexts
- DL/OWL compatibility
- Probabilistic overlays

## Performance Strategy

VSAR addresses classical reasoner blowups through:

1. **Approximate joins by vector retrieval** (vs relational join enumeration)
2. **Beam search + thresholds** (only top-k candidates explored)
3. **Vectorized execution** (batch compute similarities, GPU support)
4. **Predicate partitioning** (separate bundles per predicate)
5. **Caching and semi-naive evaluation** (memoize subquery results)
6. **Novelty detection** (avoid reinserting near-duplicate derived facts)

## Documentation

### Structure
- Overview: philosophy, approximate reasoning, VSA vs classical logic
- Language reference: full VSARL syntax
- Formal semantics: readable version of specification §10
- Tutorials: step-by-step reasoning examples
- User guide: common patterns, performance tuning
- API reference: Python classes and methods

### Learning Path
1. Hello world: facts + queries
2. Conjunctive queries
3. Rules and chaining
4. Approximation controls
5. Traces and explanations
6. Clifford mode (advanced)

## Simplicity Principles

- Make every change as simple as possible
- Avoid massive or complex changes
- Impact as little code as possible
- No over-engineering: only add what's directly requested
- Don't add features, refactors, or "improvements" beyond what was asked
- Don't add error handling for scenarios that can't happen
- Don't create helpers/utilities/abstractions for one-time operations
