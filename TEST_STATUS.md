# Test Status and Coverage Issues

## Current Status

**Test Results**: 610 tests collected, **~90 failures** across multiple modules
**Coverage**: **6.28%** (Required: 90%)

## Issues to Fix

### 1. Low Coverage (Priority: HIGH)

Many new modules have 0% coverage:

- `src/vsar/reasoning/` - Backward chaining, NAF, stratification
- `src/vsar/store/` - Belief state, fact store, item schema
- `src/vsar/unification/` - Decoder, substitution, unifier
- `src/vsar/encoding/atom_encoder.py` - Role-filler encoder
- `src/vsar/symbols/codebook.py` - Codebook implementation

**Root Cause**: New features were added with backend tests but not integrated into the main coverage suite.

**Solution**:
1. Ensure all new modules are imported in tests
2. Add integration tests that exercise the new code paths
3. Consider whether some modules should be excluded from coverage (e.g., experimental features)

### 2. Test Failures by Module

#### Fixed
- ✅ `tests/unit/kb/test_persistence.py` - Fixed typo: `dimensionension` → `dimension`

#### To Fix

**tests/encoding/test_atom_encoding.py** (3 failures)
- Issues with atom encoder integration

**tests/integration/test_vsa_flow.py** (7 failures)
- VSA flow integration broken

**tests/negation/test_classical_negation.py** (2 failures)
- Classical negation edge cases

**tests/reasoning/test_query.py** (4 failures)
- Query engine integration issues

**tests/unit/kb/test_store.py** (18 failures)
- Knowledge base store API changes

**tests/unit/kernel/test_vsa_backend.py** (3 failures)
- Backend API changes

**tests/unit/retrieval/** (Multiple failures)
- Cleanup, query, unbind modules

**tests/unit/symbols/** (Multiple failures)
- Basis, registry, spaces modules

## Pre-Commit Hook Strategy

To prevent CI failures while allowing development:

### On Commit (Fast)
- ✅ Code formatting (black)
- ✅ Linting (ruff)
- ✅ Type checking (mypy)
- ✅ Quick sanity tests (language + CLI modules)

### On Push (Thorough)
- ✅ Full test suite with coverage requirement
- ✅ Catches issues before CI runs

## Running Tests Locally

```bash
# Quick sanity check (runs on commit)
./scripts/run_tests.sh quick

# Unit tests only
./scripts/run_tests.sh unit

# Integration tests only
./scripts/run_tests.sh integration

# Generate coverage report
./scripts/run_tests.sh coverage

# Full CI test (runs on push)
./scripts/run_tests.sh ci

# Re-run only failed tests
./scripts/run_tests.sh failed
```

## Next Steps

1. **Immediate**: Fix failing tests in critical modules
   - `test_vsa_flow.py` - Core VSA integration
   - `test_store.py` - Knowledge base operations
   - `test_query.py` - Query execution

2. **Short-term**: Increase coverage
   - Add tests for `reasoning/` module
   - Add tests for `store/` module
   - Add tests for `unification/` module

3. **Long-term**: Maintain >90% coverage
   - Add tests for all new features
   - Integrate coverage checks in development workflow
   - Consider excluding experimental/GUI code

## Coverage Exclusions

Already excluded from coverage (see `pyproject.toml`):
- `*/tests/*` - Test files themselves
- `*/version.py` - Version string
- `*/cli/main.py` - CLI entry point (tested via integration)
- `*/ide/*.py` - GUI code (tested manually)

Consider adding:
- Experimental modules
- Deprecated code
- Type stubs

## CI Configuration

The GitHub Actions CI runs:
```bash
uv run pytest --cov=vsar --cov-fail-under=90
```

This will fail until:
1. Test failures are fixed
2. Coverage reaches 90%
