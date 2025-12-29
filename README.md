# VSAR: VSA-grounded Reasoning

VSAR (VSAX Reasoner) is a VSA-grounded reasoning tool that provides a unified inference substrate using hypervector algebra for deductive reasoning, approximate joins, and explainable results.

Built on top of the [VSAX library](https://vsarathy.com/vsax/) for GPU-accelerated VSA operations.

## Key Features

- **VSA-grounded reasoning**: Leverage hypervector algebra for approximate unification
- **Predicate-partitioned storage**: Efficient KB organization reduces retrieval noise
- **Top-k retrieval**: Similarity-based nearest neighbor search for variable bindings
- **Deterministic generation**: Reproducible results with fixed seeds
- **HDF5 persistence**: Save and load knowledge bases and symbol registries
- **Comprehensive testing**: 179 tests with 99% coverage

## Quick Start

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

# Insert facts: parent(alice, bob), parent(bob, carol)
facts = [("alice", "bob"), ("bob", "carol")]
for args in facts:
    atom_vec = encoder.encode_atom("parent", list(args))
    kb.insert("parent", atom_vec, args)

# Query: parent(alice, X)
results = retriever.retrieve("parent", 2, {"1": "alice"}, k=5)
print(results)  # [('bob', 0.85), ...]

# Query: parent(X, carol)
results = retriever.retrieve("parent", 1, {"2": "carol"}, k=5)
print(results)  # [('bob', 0.78), ...]
```

## Installation

### Using uv (recommended)

```bash
# Install uv
pip install uv

# Clone and install from source
git clone https://github.com/your-org/vsar.git
cd vsar
uv sync

# Run tests
uv run pytest
```

### Using pip (future)

```bash
pip install vsar
```

## Development

```bash
# Clone the repository
git clone https://github.com/your-org/vsar.git
cd vsar

# Install dependencies
uv sync

# Run tests with coverage
uv run pytest --cov=vsar --cov-fail-under=90

# Format code
uv run black .

# Lint code
uv run ruff check .

# Type check
uv run mypy src/vsar

# Run pre-commit hooks
uv run pre-commit run --all-files

# Build package
uv build
```

## Architecture

VSAR is organized into layers:

- **Kernel Layer** (`vsar.kernel`): VSA operations (FHRR backend via VSAX)
- **Symbol Layer** (`vsar.symbols`): Typed symbol spaces (E, R, A, C, T, S) with basis persistence
- **Encoding Layer** (`vsar.encoding`): Role-filler binding for atoms and queries
- **KB Layer** (`vsar.kb`): Predicate-partitioned storage with HDF5 persistence
- **Retrieval Layer** (`vsar.retrieval`): Unbinding, cleanup, and top-k retrieval

See [docs/architecture.md](docs/architecture.md) for details.

## Project Status

**Phase 0 (Foundation)** - ✅ **COMPLETE**
- ✅ Kernel backend (FHRR VSA via VSAX)
- ✅ Symbol space management (6 typed spaces)
- ✅ Atom encoding (role-filler binding)
- ✅ KB storage (predicate-partitioned bundles)
- ✅ Retrieval primitive (unbind → cleanup)
- ✅ Comprehensive tests (179 tests, 99% coverage)
- ✅ Integration tests (end-to-end workflows)
- ✅ HDF5 persistence (KB + basis)

**Future Phases**:
- Phase 1: VSARL language, query compiler, CLI
- Phase 2: Rule engine, forward chaining
- Phase 3: Optimizations, indexing

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api/)
- [CLAUDE.md](CLAUDE.md) - Developer guide

## Testing

VSAR has comprehensive test coverage:

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=vsar --cov-report=html

# Run specific test suites
uv run pytest tests/unit/           # Unit tests
uv run pytest tests/integration/    # Integration tests
```

Test statistics:
- **179 tests** (all passing)
- **99.07% coverage**
- Unit tests: 156
- Integration tests: 23

## License

MIT