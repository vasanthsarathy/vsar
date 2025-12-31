# Changelog

All notable changes to VSAR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-31

### Added - Phase 2: Horn Rules & Chaining

**Major Features:**
- Horn clause rules with `head :- body1, body2, ...` syntax
- Forward chaining with fixpoint detection
- Semi-naive evaluation optimization
- Novelty detection via similarity threshold
- Query with automatic rule application
- Multi-hop inference (transitive closure)
- Recursive rules support
- Beam search joins for multi-body rules

**New Modules:**
- `vsar.semantics.substitution` - Variable binding management
- `vsar.semantics.join` - Beam search join operations
- `vsar.semantics.chaining` - Forward chaining engine

**Extended Modules:**
- Extended `VSAREngine.query()` with optional `rules` parameter
- Extended `VSAREngine.apply_rule()` with novelty checking
- Extended `KnowledgeBase.contains_similar()` for duplicate detection
- Extended `Retriever` to support queries with no bound arguments

**Examples & Documentation:**
- Added 6 example VSAR programs demonstrating Phase 2 features:
  - `examples/01_basic_rules.vsar` - Simple rule derivation
  - `examples/02_family_tree.vsar` - Multi-hop grandparent inference
  - `examples/03_transitive_closure.vsar` - Recursive ancestor rules
  - `examples/04_organizational_hierarchy.vsar` - Manager chains
  - `examples/05_knowledge_graph.vsar` - Multi-relation connections
  - `examples/06_academic_network.vsar` - Complex multi-rule interactions
- Added `examples/README.md` with comprehensive usage guide
- Added `PROGRESS.md` - Detailed capability analysis and roadmap
- Added end-to-end integration tests in `tests/integration/test_e2e_phase2.py`
- Updated `CLAUDE.md` with Phase 2 completion details
- Updated `README.md` with Phase 2 features and examples

**Testing:**
- Added 88 new tests (Phase 2)
- Total: 392 tests passing (4 skipped)
- Coverage: 97.56% (up from 98.5%)
- New test files:
  - `tests/unit/semantics/test_substitution.py` (27 tests)
  - `tests/unit/semantics/test_join.py` (14 tests)
  - `tests/integration/test_chaining.py` (17 tests)
  - `tests/integration/test_query_with_rules.py` (12 tests)
  - `tests/integration/test_e2e_phase2.py` (5 tests)

### Changed

- Updated query execution to optionally run forward chaining before retrieval
- Improved trace collection to include chaining events
- Enhanced KB statistics to include derived predicates
- Updated README with comprehensive Phase 2 documentation
- Reorganized documentation structure

### Technical Details

**Variable Substitution:**
- Immutable substitution bindings
- Compose multiple substitutions
- Apply substitutions to atoms
- Utility functions for variable detection

**Join Operations:**
- CandidateBinding class for partial variable bindings
- Beam search to limit combinatorial explosion
- Score propagation through joins (approximate joint probability)
- Support for both single and multi-variable atoms

**Forward Chaining:**
- Iterative rule application until fixpoint
- Configurable max iterations (default: 100)
- Tracks derived facts per iteration
- Semi-naive optimization (tracks new predicates per iteration)
- Returns ChainingResult with detailed statistics

**Novelty Detection:**
- Cosine similarity threshold (default: 0.95)
- Prevents duplicate derived facts
- Configurable via `@novelty` directive
- Aligns with VSA's approximate nature

### Performance Improvements

- Semi-naive evaluation significantly faster than naive re-evaluation
- Only processes rules when body predicates have new facts
- Beam search prevents exponential blowup in joins
- Vectorized operations maintained throughout

### Known Limitations

- Single-variable queries only (multi-variable planned for Phase 3)
- No negation support yet
- No aggregation (count, sum, max)
- Forward chaining only (no backward chaining)
- No magic sets optimization

## [0.2.3] - 2024-12-29

### Fixed
- Show detailed error messages in CLI instead of generic "Error processing program"
- Improved error reporting for parser and loader errors
- Better user experience for debugging VSAR programs

## [0.2.2] - 2024-12-29

### Fixed
- Resolve path issues on Windows by converting all paths to absolute paths
- Fix grammar file loading across different operating systems
- Improve cross-platform compatibility

## [0.2.1] - 2024-12-28

### Fixed
- Fix packaging to include `grammar.lark` file in distribution
- Ensure grammar file is properly installed with package

## [0.2.0] - 2024-12-28

### Added - Phase 1: Language & CLI

**Language Features:**
- VSARL parser with Lark grammar
- AST representation (Program, Directive, Fact, Query)
- Support for facts, queries, and directives
- Comments (single-line `//` and multi-line `/* */`)
- Program loader for `.vsar` files

**CLI Interface:**
- `vsar run` - Execute VSAR programs
- `vsar ingest` - Load facts from CSV/JSONL
- `vsar export` - Export KB to JSON/JSONL
- `vsar inspect` - Show KB statistics
- `vsar repl` - Interactive REPL
- Rich formatting for tables and output

**Trace Layer:**
- Trace collection with DAG structure
- Event types: query, retrieval, fact_insertion
- Parent-child relationships
- Payload data for debugging

**File Format Support:**
- CSV ingestion (with/without predicate column)
- JSONL ingestion/export
- Native `.vsar` program files
- HDF5 persistence (from Phase 0)

**Testing:**
- 281 tests with 98.5% coverage
- Unit tests for parser, loader, CLI
- Integration tests for full program execution
- Fixed seeds for deterministic testing

### Changed
- Refactored engine to use AST types
- Improved error messages and validation
- Better documentation and examples

## [0.1.0] - 2024-12-15

### Added - Phase 0: Foundation

**Core VSA System:**
- FHRR backend via VSAX library
- MAP backend support (alternative)
- JAX-based vectorized operations
- Configurable dimensionality and seeds

**Symbol Management:**
- SymbolRegistry with typed symbol spaces
- 6 symbol spaces: E (entities), R (relations), A (attributes), C (contexts), T (time), S (structural)
- Basis vector management
- Deterministic symbol encoding

**Atom Encoding:**
- Role-filler binding for atoms
- VSAEncoder with predicate + argument encoding
- RoleVectorManager for role assignment
- Support for arbitrary arity facts

**Knowledge Base:**
- Predicate-partitioned storage
- Bundle-based fact representation
- Efficient insertion and retrieval
- HDF5 persistence (save/load)

**Retrieval System:**
- Unbinding operation for variable extraction
- Cleanup via similarity search
- Top-k retrieval with scores
- Support for single-variable queries

**Testing:**
- Comprehensive unit test coverage
- Integration tests for full pipeline
- Deterministic testing with fixed seeds
- 150+ tests

### Technical Foundation
- Modular architecture (kernel → symbols → encoding → KB → retrieval)
- Type-safe Python with Pydantic models
- JAX backend for GPU acceleration
- HDF5 for efficient persistence

---

## Version History Summary

- **0.3.0** (2025-12-31) - Phase 2: Horn Rules & Chaining ✓
- **0.2.3** (2024-12-29) - Bug fix: Better error messages
- **0.2.2** (2024-12-29) - Bug fix: Windows path resolution
- **0.2.1** (2024-12-28) - Bug fix: Grammar file packaging
- **0.2.0** (2024-12-28) - Phase 1: Language & CLI ✓
- **0.1.0** (2024-12-15) - Phase 0: Foundation ✓

---

## Upgrade Guide

### Upgrading from 0.2.x to 0.3.0

**New Features Available:**
- Define rules in your VSAR programs using `rule head :- body1, body2.` syntax
- Query with automatic rule application: `engine.query(query, rules=rules)`
- Forward chaining: `from vsar.semantics.chaining import apply_rules`
- Configure beam width: `@beam 50;`
- Configure novelty threshold: `@novelty 0.95;`

**Breaking Changes:**
None - fully backward compatible

**Recommended Actions:**
1. Review example programs in `examples/` directory
2. Read `PROGRESS.md` for capability analysis
3. Try transitive closure queries (see `examples/03_transitive_closure.vsar`)

### Upgrading from 0.1.x to 0.2.0

**New Features Available:**
- VSARL language syntax
- CLI commands (`vsar run`, `vsar ingest`, etc.)
- Interactive REPL
- Trace collection

**Breaking Changes:**
- Engine initialization now uses `Directive` objects instead of raw params
- Query execution uses `Query` AST objects

**Migration Example:**
```python
# Old (0.1.x)
engine = VSAREngine(backend_type="FHRR", dim=512, seed=42)
results = engine.query("parent", var_position=2, bound_args={"1": "alice"})

# New (0.2.0+)
from vsar.language.ast import Directive, Query
directives = [Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})]
engine = VSAREngine(directives)
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query)
```

---

## Future Roadmap

### Phase 3: Advanced Features (Planned)
- Multi-variable queries
- Stratified negation
- Aggregation operators
- Backward chaining
- Magic sets optimization

### Phase 4: Scale & Performance (Planned)
- Incremental maintenance
- Query planning
- Parallel execution
- GPU acceleration for large KBs

### Phase 5: Advanced Reasoning (Research)
- Probabilistic reasoning
- Temporal reasoning
- Defeasible reasoning
- Ontology integration (OWL/DL)

---

For detailed technical information, see:
- [PROGRESS.md](PROGRESS.md) - Current capabilities and limitations
- [CLAUDE.md](CLAUDE.md) - Development workflow and architecture
- [README.md](README.md) - Getting started and examples
