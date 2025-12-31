# Changelog

For the complete changelog, see [CHANGELOG.md](https://github.com/vasanthsarathy/vsar/blob/main/CHANGELOG.md) in the repository.

## Latest Release: v0.3.0 (2025-12-31)

### Phase 2: Horn Rules & Forward Chaining

Major feature release adding full Horn clause reasoning with forward chaining.

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

**Testing:**
- 88 new tests (Phase 2)
- Total: 392 tests passing (4 skipped)
- Coverage: 97.56%

**Breaking Changes:** None - fully backward compatible

See [full changelog](https://github.com/vasanthsarathy/vsar/blob/main/CHANGELOG.md) for details.

## Previous Releases

- **0.2.3** (2024-12-29) - Bug fix: Better error messages
- **0.2.2** (2024-12-29) - Bug fix: Windows path resolution
- **0.2.1** (2024-12-28) - Bug fix: Grammar file packaging
- **0.2.0** (2024-12-28) - Phase 1: Language & CLI
- **0.1.0** (2024-12-15) - Phase 0: Foundation

## Upgrade Guides

### From 0.2.x to 0.3.0

**New Features:**
- Define rules using `rule head :- body1, body2.` syntax
- Query with automatic rule application
- Forward chaining API
- Beam width and novelty configuration

**No Breaking Changes** - Fully backward compatible

**Migration:** None required

### From 0.1.x to 0.2.0

**New Features:**
- VSARL language syntax
- CLI commands
- Interactive REPL

**Breaking Changes:**
- Engine initialization uses `Directive` objects
- Query execution uses `Query` AST objects

See [full changelog](https://github.com/vasanthsarathy/vsar/blob/main/CHANGELOG.md#upgrading-from-01x-to-020) for migration guide.
