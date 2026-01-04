# VSAR Multi-Mode Reasoning Implementation Plan

**Status**: Active Plan (Jan 2026)
**Current Phase**: Option C - Quick Wins for Paper (2-3 weeks)

---

## Executive Summary

**Current Status**: VSAR is production-ready for **deductive reasoning** (Horn clauses, forward chaining) with approximate matching via VSA, but lacks 11+ advanced reasoning modes envisioned in the specifications.

**Immediate Goal (Option C)**: Implement multi-variable queries and backward chaining to strengthen the IJCAI 2026 paper submission (deadline: Jan 19, 2026).

**Long-term Goal**: Build architectural foundations to enable all 11+ reasoning modes specified in the design documents.

**Key Finding**: The bottleneck is **not the encoding** (which is excellent) but the **reasoning layer**, which lacks abstraction for pluggable reasoning strategies.

---

## Current State Assessment

### ✅ IMPLEMENTED (Production Ready)

#### 1. **Deductive Reasoning** (Phase 2 Complete)
- **Status**: 387 tests passing, 97.56% coverage
- **Capabilities**:
  - Horn clause rules: `head :- body1, body2, ...`
  - Forward chaining with semi-naive evaluation
  - Multi-hop inference (arbitrary depth transitive closure)
  - Beam search joins for multi-body rules
  - Novelty detection and fixpoint detection
  - Full trace collection and provenance tracking
- **Files**:
  - `src/vsar/semantics/chaining.py` - Forward chaining engine
  - `src/vsar/semantics/join.py` - Beam search joins
  - `src/vsar/semantics/substitution.py` - Variable binding
- **Examples**: All 6 example programs in `examples/` directory

#### 2. **Negation** (Partial)
- **What works**:
  - Classical negation: `fact ~enemy(alice, bob).`
  - Negation-as-failure (NAF): `not enemy(X, _)` in rule bodies
  - Threshold-based NAF semantics
  - Stratification analysis (warns but doesn't enforce)
- **What's missing**: Stratified evaluation enforcement
- **Files**: `src/vsar/reasoning/naf.py`, `stratification.py`, `consistency.py`

#### 3. **Approximate Matching** (Core Differentiator)
- Similarity-based retrieval instead of exact unification
- Ranked results with confidence scores
- Graceful degradation under noise
- Configurable thresholds via `@threshold`, `@beam`, `@novelty`

#### 4. **Explainability & Tracing**
- Full execution trace collection
- Provenance tracking
- Trace DAG construction and display

### 🔧 INFRASTRUCTURE PREPARED (Not Integrated)

- **Paraconsistent Belief Tracking**: `src/vsar/store/belief.py` - Code exists, not hooked into engine
- **Extended Item Metadata**: `ItemKind` enum with `AXIOM`, `EDGE`, `CASE`, `MAP` (unused)
- Fields for `weight`, `priority`, `agent`, `provenance` (prepared for future modes)

### ❌ NOT IMPLEMENTED (Specified But Missing)

1. **Multi-variable queries** - `parent(?, ?)` ← **OPTION C TARGET**
2. **Backward chaining** - Goal-directed search ← **OPTION C TARGET**
3. **Aggregation** - `count()`, `sum()`, `max()`, `min()`
4. **Description Logic (DL)** - ALC tableau reasoning
5. **Default reasoning** - Defeasible rules with priorities
6. **Probabilistic reasoning** - Weighted propagation, uncertainty
7. **Argumentation** - Support/attack graphs
8. **Epistemic reasoning** - Multi-agent KBs
9. **Abductive reasoning** - Hypothesis generation
10. **Analogical reasoning** - Structure mapping
11. **Case-based reasoning** - Case retrieval and adaptation
12. **Inductive learning** - Rule template generation

---

## Option C: Quick Wins for Paper (ACTIVE - 2-3 weeks)

### Goal
Strengthen IJCAI 2026 paper by:
1. Addressing reviewer feedback on "limited expressivity"
2. Demonstrating VSAR is more than just forward chaining
3. Adding multi-variable query capability (directly requested)

### Timeline

**Week 1: Multi-Variable Queries (Jan 6-10)**
- Remove single-variable restriction in `src/vsar/semantics/engine.py`
- Implement iterative beam search algorithm in `src/vsar/retrieval/query.py`
- Add tests for `parent(?, ?)` queries
- Benchmark accuracy vs single-variable queries

**Week 2: Backward Chaining (Jan 13-17)**
- Implement SLD resolution in `src/vsar/reasoning/backward_chaining.py`
- Add tabling (memoization) to avoid infinite loops
- Integrate with VSAREngine
- Comparative benchmarking vs forward chaining

**Week 3: Paper Updates (Jan 18-19)**
- Add experimental results for multi-variable queries
- Add comparison of forward vs backward chaining
- Update "Limitations and Extensions" section
- Brief mention of architectural extensibility

### Deliverables

1. **Code**:
   - Multi-variable query support
   - Backward chaining implementation
   - Comprehensive tests

2. **Paper Updates**:
   - New experimental section on multi-variable queries
   - Backward vs forward chaining comparison
   - Updated future work section

3. **Documentation**:
   - Updated README with new capabilities
   - Example programs demonstrating new features

---

## Multi-Variable Query Implementation Details

### Algorithm (Iterative Beam Search)

```python
def query_multi_variable(predicate, query_pattern):
    """
    Handle queries like parent(?, ?) with multiple unknowns.

    Algorithm:
    1. Unbind predicate for each fact
    2. Decode position 1, retrieve top-B candidates
    3. For each candidate, cancel its contribution
    4. Decode position 2, retrieve top entity
    5. Rank pairs by joint similarity
    """
    results = []

    for fact in kb.get_facts(predicate):
        # Unbind predicate
        arg_bundle = unbind(fact, predicate_vector)

        # Decode first variable position
        q1 = shift(arg_bundle, -1)
        candidates_pos1 = cleanup(q1, beam_width=B)

        # For each candidate, decode second position
        for cand1, sim1 in candidates_pos1:
            # Cancel first position contribution
            cleaned = arg_bundle - shift(cand1.vector, 1)

            # Decode second position
            q2 = shift(cleaned, -2)
            cand2, sim2 = cleanup(q2, top_k=1)

            # Joint similarity score
            joint_sim = (sim1 + sim2) / 2
            results.append((cand1, cand2, joint_sim))

    # Return top-k by joint similarity
    return sorted(results, key=lambda x: x[2], reverse=True)[:k]
```

### Code Changes

**File**: `src/vsar/semantics/engine.py`
```python
# REMOVE THIS CHECK:
if len(var_positions) != 1:
    raise ValueError(f"Query must have exactly 1 variable, got {len(var_positions)}")

# REPLACE WITH:
if len(var_positions) == 1:
    return self._query_single_variable(...)
elif len(var_positions) >= 2:
    return self._query_multi_variable(...)
```

**File**: `src/vsar/retrieval/query.py`
- Add `_query_multi_variable()` method implementing iterative beam search

### Testing

```python
def test_multi_variable_query():
    """Test parent(?, ?) returns all parent-child pairs."""
    kb = KnowledgeBase()
    kb.add_fact("parent(alice, bob)")
    kb.add_fact("parent(alice, charlie)")
    kb.add_fact("parent(david, eve)")

    results = kb.query("parent(?, ?)")

    assert len(results) == 3
    assert ("alice", "bob") in [(r[0], r[1]) for r in results]
    assert ("alice", "charlie") in [(r[0], r[1]) for r in results]
    assert ("david", "eve") in [(r[0], r[1]) for r in results]
```

---

## Backward Chaining Implementation Details

### Algorithm (SLD Resolution with Tabling)

```python
class BackwardChainingStrategy(ReasoningStrategy):
    """Goal-directed proof search using SLD resolution."""

    def __init__(self, kb, max_depth=10):
        self.kb = kb
        self.max_depth = max_depth
        self.table = {}  # Memoization cache

    def prove_goal(self, goal, depth=0):
        """
        Prove goal using SLD resolution.

        Returns: List of (substitution, similarity) pairs
        """
        if depth >= self.max_depth:
            return []

        # Check table (memoization)
        if goal in self.table:
            return self.table[goal]

        results = []

        # Try to match against facts
        for fact in self.kb.get_facts(goal.predicate):
            theta, sim = unify_vsa(goal, fact)
            if sim > threshold:
                results.append((theta, sim))

        # Try to match against rules
        for rule in self.kb.get_rules_with_head(goal.predicate):
            # Unify goal with rule head
            theta, sim = unify_vsa(goal, rule.head)
            if sim > threshold:
                # Recursively prove body
                body_results = self.prove_conjunction(
                    [apply_substitution(lit, theta) for lit in rule.body],
                    depth + 1
                )
                for body_theta, body_sim in body_results:
                    # Combine substitutions
                    combined = compose_substitutions(theta, body_theta)
                    combined_sim = min(sim, body_sim)
                    results.append((combined, combined_sim))

        # Store in table
        self.table[goal] = results
        return results

    def prove_conjunction(self, goals, depth):
        """Prove conjunction of goals (AND)."""
        if not goals:
            return [({}, 1.0)]  # Empty conjunction is trivially true

        first, *rest = goals
        results = []

        # Prove first goal
        for theta1, sim1 in self.prove_goal(first, depth):
            # Apply substitution to remaining goals
            remaining = [apply_substitution(g, theta1) for g in rest]

            # Prove remaining goals
            for theta2, sim2 in self.prove_conjunction(remaining, depth):
                combined = compose_substitutions(theta1, theta2)
                combined_sim = min(sim1, sim2)
                results.append((combined, combined_sim))

        return results
```

### Key Challenges

1. **Approximate Unification**: VSA unification is approximate, not exact
   - Solution: Use beam search to explore multiple candidates
   - Threshold management to prune low-similarity branches

2. **Tabling with Approximate Results**: Caching approximate proofs
   - Solution: Cache top-k results per goal
   - Invalidate cache on KB updates

3. **Infinite Loops**: Recursive rules can cause infinite descent
   - Solution: Depth limit + cycle detection
   - Track goal stack to detect cycles

### Testing

```python
def test_backward_chaining_transitive_closure():
    """Compare backward vs forward chaining on ancestor queries."""
    kb = KnowledgeBase()
    kb.add_fact("parent(alice, bob)")
    kb.add_fact("parent(bob, charlie)")
    kb.add_rule("ancestor(X, Z) :- parent(X, Z).")
    kb.add_rule("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).")

    # Forward chaining: derive all ancestors
    forward_results = kb.query("ancestor(alice, ?)", strategy="forward")

    # Backward chaining: goal-directed search
    backward_results = kb.query("ancestor(alice, charlie)?", strategy="backward")

    # Should get same results
    assert set(forward_results) == set(backward_results)
```

---

## After Paper Submission: Long-Term Roadmap

### Phase A: Architectural Foundations (4-6 weeks)

**Goal**: Create extension points for all reasoning modes

1. **ReasoningStrategy ABC** (1 week)
   - Abstract interface for pluggable strategies
   - Refactor existing code into `ForwardChainingStrategy`
   - Enable dynamic strategy selection

2. **Encoder Registry** (3-5 days)
   - Plugin system for encoders
   - `@encoding <name>` directive support

3. **Configuration System** (1 week)
   - `@reasoning`, `@semantics`, `@mode` directives
   - Engine configuration propagation

4. **Paraconsistent Integration** (1 week)
   - Hook belief state into engine
   - 4-valued logic support

### Phase B: High-Value Modes (8-12 weeks)

1. **Aggregation** (1-2 weeks) - `count()`, `sum()`, `max()`, `min()`
2. **Abduction** (3-4 weeks) - Hypothesis generation (requires FormulaEncoder)
3. **Analogical** (4-5 weeks) - Structure mapping (requires FormulaEncoder)

### Phase C: Advanced Modes (12+ weeks)

1. **Description Logic** (3-4 weeks) - ALC tableau
2. **Probabilistic** (2-3 weeks) - Weighted propagation
3. **Argumentation** (3-4 weeks) - Support/attack graphs
4. **Case-Based** (2-3 weeks) - Case retrieval and adaptation
5. **Inductive Learning** (3-4 weeks) - Rule learning

---

## Implementation Priority Matrix

| Mode | Priority | Effort | Prerequisites | Status |
|------|----------|--------|---------------|--------|
| **Multi-variable queries** | CRITICAL | 1 week | None | ← IN PROGRESS |
| **Backward chaining** | CRITICAL | 2 weeks | None | ← NEXT |
| **ReasoningStrategy ABC** | HIGH | 1 week | None | After paper |
| **Encoder Registry** | HIGH | 3-5 days | None | After paper |
| **Configuration System** | HIGH | 1 week | None | After paper |
| **Paraconsistent integration** | HIGH | 1 week | None | After paper |
| **Aggregation** | MEDIUM-HIGH | 1-2 weeks | None | Phase B |
| **Abduction** | MEDIUM | 3-4 weeks | FormulaEncoder | Phase B |
| **Analogical** | MEDIUM | 4-5 weeks | FormulaEncoder | Phase B |
| **Description Logic** | MEDIUM | 3-4 weeks | ReasoningStrategy | Phase C |
| **Probabilistic** | MEDIUM | 2-3 weeks | None | Phase C |
| **Argumentation** | LOW | 3-4 weeks | None | Phase C |
| **Case-based** | LOW | 2-3 weeks | Analogy | Phase C |
| **Inductive learning** | LOW | 3-4 weeks | FormulaEncoder | Phase C |

---

## Critical Files Reference

### Currently Implemented

**Deductive**:
- `src/vsar/semantics/chaining.py` - Forward chaining
- `src/vsar/semantics/join.py` - Beam search joins
- `src/vsar/semantics/substitution.py` - Variable binding

**Negation**:
- `src/vsar/reasoning/naf.py`
- `src/vsar/reasoning/stratification.py`
- `src/vsar/reasoning/consistency.py`

**Infrastructure**:
- `src/vsar/store/belief.py` - Paraconsistent (not integrated)
- `src/vsar/store/item.py` - Extended metadata
- `src/vsar/symbols/spaces.py` - Symbol spaces

**Encoding**:
- `src/vsar/encoding/vsa_encoder.py` - Shift-based
- `src/vsar/encoding/role_filler_encoder.py` - Hybrid (CURRENT)
- `src/vsar/encoding/atom_encoder.py` - Bind-based (experimental)

**Kernel**:
- `src/vsar/kernel/vsa_backend.py` - FHRR and MAP
- `src/vsar/kernel/base.py` - KernelBackend ABC

### To Create (Option C - This Week!)

- `src/vsar/retrieval/multi_variable.py` - Multi-variable query logic
- `src/vsar/reasoning/backward_chaining.py` - Backward chaining
- `tests/integration/test_multi_variable.py` - Tests
- `tests/integration/test_backward_chaining.py` - Tests

### To Create (Phase A - After Paper)

- `src/vsar/reasoning/strategy.py` - ReasoningStrategy ABC
- `src/vsar/encoding/registry.py` - EncoderRegistry
- `src/vsar/config/directive_parser.py` - Config parsing
- `src/vsar/config/config_manager.py` - Config management

---

## Success Metrics

### For Option C (Paper Deadline)

**Must Have**:
- ✅ Multi-variable queries working (`parent(?, ?)`)
- ✅ Backward chaining implementation
- ✅ Tests passing for both features
- ✅ Paper updated with new capabilities

**Nice to Have**:
- Performance benchmarks showing backward chaining efficiency
- Comparison table: forward vs backward chaining
- Example programs demonstrating new features

### For Phase A (Post-Paper)

- ✅ All 387 existing tests still passing
- ✅ ReasoningStrategy abstraction in place
- ✅ At least 2 strategies implemented (forward, backward)
- ✅ Configuration system working

---

## Notes

- **Specification Coverage**: This plan addresses ~90% gap between 11-mode specification and current 1-mode implementation
- **Architectural Soundness**: Kernel layer is excellent (9/10), reasoning layer needs refactor (5/10)
- **Quick Win Strategy**: Option C addresses paper reviewers' concerns while building toward long-term vision
- **Realistic Timelines**: Estimates based on codebase complexity and test coverage requirements

---

## Related Documents

- **Specification PDFs** (in `spec/`):
  - `Vsar Inference Engine_ Unified Multi-mode Control Semantics.pdf`
  - `Vsar Unified Encoding & Reasoning Framework (clean Edition).pdf`
  - `Vsar-dsl_ A Specification Language For Multi-mode Vsa Reasoning Programs.pdf`

- **Design Documents** (in `docs/`):
  - `encoding-extensibility-design.md` - Future encoding architecture (not yet implemented)
  - `reasoning-algorithms.md` - Algorithm specifications for all modes

- **Legacy Planning** (archived):
  - Previous `IMPLEMENTATION_PLAN.md` superseded by this document

---

**Last Updated**: January 4, 2026
**Status**: Active - Option C in progress
**Next Milestone**: Multi-variable queries (Week 1)
