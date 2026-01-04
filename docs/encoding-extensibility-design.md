# Encoding Extensibility Design

**Version**: 0.1 (Draft)
**Date**: 2025-12-31
**Status**: Design Phase

## Executive Summary

This document proposes a comprehensive encoding extensibility system for VSAR, allowing users to define custom encoding strategies for all semantic elements: constants, predicates, atoms, rules, queries, and composite structures. The goal is to provide maximum flexibility for users to experiment with different VSA encoding approaches while maintaining backward compatibility.

## Motivation

Currently, VSAR hardcodes encoding decisions:
- **Constants** → basis vectors in typed symbol spaces
- **Predicates** → bundled with arguments via circular permutation
- **Atoms** → `shift(hv(t1), 1) ⊕ shift(hv(t2), 2) ⊕ ...`
- **Rules** → not directly encoded (evaluated procedurally)

This limits experimentation with alternative encoding strategies. Users should be able to specify:
- How to encode positional information (shift vs role-filler vs other)
- How to combine predicate with arguments
- How to represent structured terms
- How to encode negation, defaults, and other logical constructs

## What Needs Encoding?

### 1. Constants (Symbols)

**Current**: Registered in typed symbol spaces, assigned basis vectors
```python
entity_vec = registry.register(SymbolSpace.ENTITIES, "alice")
```

**Encoding Decisions**:
- Which symbol space to use (ENTITIES, RELATIONS, ATTRIBUTES, etc.)
- Basis vector generation (random, deterministic, learned)
- Dimensionality (same for all spaces or per-space?)
- Normalization strategy

**User Control Points**:
```vsarl
@symbol_encoding {
    basis: "random",        // random, deterministic, orthogonal, learned
    seed: 42,
    normalize: true,
    space_strategy: "typed" // typed (current), unified, per-predicate
};
```

### 2. Predicates (Relations)

**Current**: Implicit - predicates not encoded separately, only used for KB partitioning

**Encoding Decisions**:
- Should predicates have their own vectors?
- How to incorporate predicate into atom encoding?
- Predicate arity encoding (fixed vs variable)

**User Control Points**:
```vsarl
@predicate_encoding {
    include_in_atom: false,  // Current: predicates not in atom vector
    arity_binding: true,     // Encode arity information
    relation_space: "R"      // Symbol space for relations
};

// Alternative: encode predicate into atom
@predicate_encoding {
    include_in_atom: true,
    composition: "bind",     // bind, bundle, concat
    position: "first"        // first, last, distributed
};
```

### 3. Atoms (Ground Facts)

**Current**: Shift-based positional encoding
```python
enc(parent(alice, bob)) = shift(hv(alice), 1) ⊕ shift(hv(bob), 2)
```

**Alternative Strategies**:

**A. Role-Filler Binding**:
```python
enc(parent(alice, bob)) = ρ1 ⊗ hv(alice) ⊕ ρ2 ⊗ hv(bob)
```

**B. Predicate Inclusion**:
```python
enc(parent(alice, bob)) = hv(parent) ⊗ (shift(hv(alice), 1) ⊕ shift(hv(bob), 2))
```

**C. Hybrid**:
```python
enc(parent(alice, bob)) =
    α * shift(hv(alice), 1) ⊕
    β * (ρ1 ⊗ hv(alice)) ⊕
    shift(hv(bob), 2) ⊕
    (ρ2 ⊗ hv(bob))
```

**User Control Points**:
```vsarl
@atom_encoding shift;                    // Simple named strategy

@atom_encoding role_filler(seed=42);     // With parameters

@atom_encoding {
    strategy: "hybrid",
    shift_weight: 0.6,
    role_weight: 0.4,
    include_predicate: true,
    normalize: true
};

// Compositional specification
@atom_encoding {
    predicate: {action: "bind", position: "first"},
    arguments: {strategy: "shift", normalize: true},
    composition: "bundle"
};
```

### 4. Variables (Logical Variables)

**Current**: Not encoded - used as placeholders in queries, substituted during retrieval

**Encoding Decisions**:
- Should variables have vectors? (for unification, etc.)
- Anonymous variables (_) vs named variables (X, Y)
- Variable scope encoding

**Possible Encodings**:
```python
# Option 1: No encoding (current - procedural substitution)
query parent(alice, X) → unbind position 2

# Option 2: Encode as special symbols
hv(X) = random_basis_vector("var_X")
enc(parent(alice, X)) = shift(hv(alice), 1) ⊕ shift(hv(X), 2)

# Option 3: Encode as "wildcard" pattern
hv(?) = wildcard_vector()
enc(parent(alice, ?)) = shift(hv(alice), 1) ⊕ shift(hv(?), 2)
```

**User Control Points**:
```vsarl
@variable_encoding {
    representation: "placeholder",  // placeholder (current), symbol, wildcard
    scope: "local",                 // local, global, scoped
    anonymous_handling: "unique"    // unique, shared, null
};
```

### 5. Terms (Complex Structures)

**Current**: Not supported - only flat atoms with constants

**Future Need**: Structured terms
```vsarl
fact author(book("War and Peace"), person(name="Tolstoy", birth=1828)).
```

**Encoding Decisions**:
- Nested structures (trees, lists, records)
- Attribute-value pairs
- Type information

**Possible Encodings**:

**A. Recursive Encoding**:
```python
enc(book("War and Peace")) =
    hv(book) ⊗ hv("War and Peace")

enc(person(name="Tolstoy", birth=1828)) =
    hv(person) ⊗ (
        hv(name) ⊗ hv("Tolstoy") ⊕
        hv(birth) ⊗ hv(1828)
    )
```

**B. Linearization**:
```python
enc(person(name="Tolstoy", birth=1828)) =
    shift(hv(person), 0) ⊕
    shift(hv(name), 1) ⊕
    shift(hv("Tolstoy"), 2) ⊕
    shift(hv(birth), 3) ⊕
    shift(hv(1828), 4)
```

**C. Graph Encoding** (Clifford algebras for orientation):
```python
enc(tree) = geometric_product_based_on_structure
```

**User Control Points**:
```vsarl
@term_encoding {
    nested: "recursive",        // recursive, linearize, graph
    attributes: "bind",         // bind, shift, bundle
    max_depth: 5,
    flatten_threshold: 10
};
```

### 6. Rules (Horn Clauses)

**Current**: Not encoded - evaluated procedurally via forward/backward chaining

**Encoding Decisions**:
- Should rules have vectors? (for meta-reasoning, rule similarity)
- Encode rule structure vs just evaluate procedurally
- Encoding substitutions and variable bindings

**Possible Encodings**:

**A. No Encoding** (current - procedural evaluation):
```python
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
# Not encoded, only executed
```

**B. Rule as Vector** (for rule similarity, analogy):
```python
enc(rule) =
    hv(head_predicate) ⊗ (
        hv(body_pred1) ⊕ hv(body_pred2)
    )
```

**C. Full Structural Encoding**:
```python
enc(rule) =
    hv(:-) ⊗ (
        enc(head_atom) ⊕
        hv(body) ⊗ (enc(body1) ⊕ enc(body2))
    )
```

**User Control Points**:
```vsarl
@rule_encoding {
    representation: "procedural",   // procedural (current), vector, hybrid
    structural: false,              // Encode rule structure
    variable_bindings: "implicit",  // implicit, explicit, vector
    similarity_metric: "predicate"  // predicate, structure, full
};

// If vector encoding enabled:
@rule_encoding {
    representation: "vector",
    head_weight: 0.6,
    body_weight: 0.4,
    variable_linking: "shared_roles"
};
```

### 7. Queries (Formulas)

**Current**: Procedurally evaluated via unbinding/retrieval

**Encoding Decisions**:
- Query as pattern vs query as vector
- How to encode query variables
- Conjunctive queries, disjunctive queries

**Possible Encodings**:

**A. Pattern-Based** (current):
```python
query parent(alice, X)?
→ unbind(enc(parent bundle), position=2)
→ retrieve top-k similar to X
```

**B. Query as Vector**:
```python
enc(parent(alice, X)) =
    shift(hv(alice), 1) ⊕ shift(hv(?), 2)
→ similarity search against all atoms
```

**C. Conjunctive Query Encoding**:
```python
query parent(X, Y), age(Y, Z)?
→ enc(query) = enc(parent(X,Y)) ⊗ enc(age(Y,Z))
```

**User Control Points**:
```vsarl
@query_encoding {
    mode: "pattern",            // pattern (current), vector, hybrid
    variable_representation: "unbind",  // unbind, wildcard, null
    conjunctive: "join",        // join (current), bundle, bind
    scoring: "factorized"       // factorized, holistic, hybrid
};
```

### 8. Negation

**Current**: Classical negation in facts (`fact !enemy(alice, bob)`), negation-as-failure in rules

**Encoding Decisions**:
- How to represent negated atoms?
- Difference between classical negation and NAF
- Strong negation vs weak negation

**Possible Encodings**:

**A. Separate Symbol Space**:
```python
enc(!enemy(alice, bob)) → store in NEGATED_FACTS space
```

**B. Negation Operator**:
```python
enc(!enemy(alice, bob)) =
    hv(¬) ⊗ enc(enemy(alice, bob))
```

**C. Phase Inversion** (for complex-valued VSAs):
```python
enc(!enemy(alice, bob)) =
    -1 * enc(enemy(alice, bob))
```

**D. Orthogonal Subspace**:
```python
enc(!enemy(alice, bob)) =
    project_to_negation_subspace(enc(enemy(alice, bob)))
```

**User Control Points**:
```vsarl
@negation_encoding {
    classical: "operator",      // operator, phase, subspace, separate
    naf: "procedural",          // procedural (current), vector
    composition: "bind",
    negation_vector: "random"   // random, learned, antipodal
};
```

### 9. Composite Structures

**Future Extensions**:

**A. Lists/Sequences**:
```vsarl
fact path([boston, newyork, philadelphia]).
```

**Encoding Options**:
```python
# Shift-based sequence
enc([a,b,c]) = shift(hv(a), 1) ⊕ shift(hv(b), 2) ⊕ shift(hv(c), 3)

# Recursive cons-cell
enc([a|rest]) = hv(cons) ⊗ (hv(a) ⊕ enc(rest))

# Trajectory encoding (for ordered sequences)
enc([a,b,c]) = geometric_sum(hv(a), hv(b), hv(c))
```

**B. Sets**:
```vsarl
fact members({alice, bob, carol}).
```

**Encoding Options**:
```python
# Unordered bundle
enc({a,b,c}) = hv(a) ⊕ hv(b) ⊕ hv(c)

# With cardinality
enc({a,b,c}) = hv(set) ⊗ (hv(|3|) ⊕ bundle(a,b,c))
```

**C. Temporal Information**:
```vsarl
fact parent(alice, bob) @ 1990.
fact location(bob, boston) @ [2020, 2025].
```

**Encoding Options**:
```python
# Time binding
enc(parent(alice,bob) @ 1990) =
    enc(parent(alice,bob)) ⊗ hv(time=1990)

# Interval encoding
enc(location @ [2020,2025]) =
    enc(location) ⊗ (hv(start=2020) ⊕ hv(end=2025))
```

**User Control Points**:
```vsarl
@composite_encoding {
    lists: "shift",              // shift, recursive, trajectory
    sets: "bundle",              // bundle, indicator, bloom
    temporal: "bind",            // bind, interval, point
    spatial: "grid"              // grid, continuous, hierarchical
};
```

### 10. Meta-Level Constructs

**A. Contexts/Worlds**:
```vsarl
fact parent(alice, bob) in world1.
fact !parent(alice, bob) in world2.
```

**Encoding**:
```python
enc(fact @ context) = hv(context) ⊗ enc(fact)
```

**B. Certainty/Probability**:
```vsarl
fact parent(alice, bob) [certainty=0.9].
```

**Encoding**:
```python
# Weight the bundle
enc(fact[c]) = c * enc(fact)

# Or bind certainty
enc(fact[c]) = enc(fact) ⊗ hv(certainty=c)
```

**C. Provenance/Source**:
```vsarl
fact parent(alice, bob) source="census".
```

**User Control Points**:
```vsarl
@meta_encoding {
    contexts: "bind",           // bind, separate_space, partition
    certainty: "weight",        // weight, bind, subspace
    provenance: "metadata",     // metadata, bind, trace
    modality: "operator"        // operator, separate, bind
};
```

## Proposed Architecture

### 1. Encoding Pipeline

Every semantic element passes through an encoding pipeline:

```
Source Element
    ↓
[Symbol Encoding] → basis vectors for constants
    ↓
[Term Encoding] → structured terms to vectors
    ↓
[Atom Encoding] → atoms to vectors
    ↓
[Composite Encoding] → rules, queries, formulas
    ↓
Final Hypervector
```

### 2. Encoder Interface Hierarchy

```python
# Base protocol
class Encoder(Protocol):
    """Base encoder interface."""
    def encode(self, element: Any, context: EncodingContext) -> jnp.ndarray: ...

# Specialized encoders
class SymbolEncoder(Encoder):
    """Encodes constants to basis vectors."""
    def encode_symbol(self, name: str, space: SymbolSpace) -> jnp.ndarray: ...

class TermEncoder(Encoder):
    """Encodes structured terms."""
    def encode_term(self, term: Term) -> jnp.ndarray: ...

class AtomEncoder(Encoder):
    """Encodes ground atoms."""
    def encode_atom(self, predicate: str, args: list[Term]) -> jnp.ndarray: ...

class FormulaEncoder(Encoder):
    """Encodes rules, queries, complex formulas."""
    def encode_rule(self, rule: Rule) -> jnp.ndarray: ...
    def encode_query(self, query: Query) -> jnp.ndarray: ...

class CompositeEncoder(Encoder):
    """Encodes lists, sets, temporal, spatial structures."""
    def encode_list(self, items: list[Term]) -> jnp.ndarray: ...
    def encode_set(self, items: set[Term]) -> jnp.ndarray: ...
    def encode_temporal(self, atom: Atom, time: TimeSpec) -> jnp.ndarray: ...
```

### 3. Encoding Context

Encoding decisions may depend on context:

```python
@dataclass
class EncodingContext:
    """Context for encoding decisions."""
    backend: KernelBackend
    registry: SymbolRegistry
    config: EncodingConfig
    parent: Optional['EncodingContext'] = None

    # Contextual information
    depth: int = 0
    scope: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 4. Encoding Configuration

All encoding decisions in one place:

```python
@dataclass
class EncodingConfig:
    """Complete encoding configuration."""

    # Symbol encoding
    symbol: SymbolEncodingConfig

    # Predicate encoding
    predicate: PredicateEncodingConfig

    # Atom encoding
    atom: AtomEncodingConfig

    # Variable encoding
    variable: VariableEncodingConfig

    # Term encoding (structured)
    term: TermEncodingConfig

    # Rule encoding
    rule: RuleEncodingConfig

    # Query encoding
    query: QueryEncodingConfig

    # Negation encoding
    negation: NegationEncodingConfig

    # Composite structures
    composite: CompositeEncodingConfig

    # Meta-level
    meta: MetaEncodingConfig
```

### 5. User Interface (VSARL Directives)

**Simple Interface** (named presets):
```vsarl
@encoding_preset "standard";     // Current default
@encoding_preset "role_filler";  // Alternative
@encoding_preset "clifford";     // Geometric algebra mode
```

**Detailed Interface** (per-component configuration):
```vsarl
@encoding {
    // Symbol encoding
    symbols: {
        basis: "random",
        normalize: true,
        seed: 42
    },

    // Atom encoding
    atoms: {
        strategy: "shift",
        include_predicate: false
    },

    // Composite structures
    composite: {
        lists: "shift",
        sets: "bundle"
    }
};
```

**Compositional Interface** (building blocks):
```vsarl
@atom_encoding shift;
@predicate_encoding separate;
@variable_encoding placeholder;
@negation_encoding operator;
```

**Maximum Flexibility** (formula-based):
```vsarl
@atom_encoding formula {
    // Mini-language for custom encoding
    predicate_vec = symbol(predicate, space="R");
    arg_vecs = [symbol(arg, space="E") for arg in args];
    shifted = [shift(v, i+1) for i, v in enumerate(arg_vecs)];
    result = bind(predicate_vec, bundle(shifted));
    return normalize(result);
};
```

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-2)
- [ ] Define encoder interface hierarchy (`Encoder`, `SymbolEncoder`, `AtomEncoder`, etc.)
- [ ] Create `EncodingConfig` dataclass hierarchy
- [ ] Implement `EncodingContext` for contextual encoding
- [ ] Refactor current `VSAEncoder` to fit new interface
- [ ] Add encoder registry for named strategies

### Phase 2: Symbol & Atom Encoders (Weeks 3-4)
- [ ] Implement `SymbolEncodingConfig` and configurable basis generation
- [ ] Implement multiple atom encoding strategies:
  - [x] Shift-based (current - refactor to use config)
  - [ ] Role-filler binding
  - [ ] Hybrid (shift + role)
  - [ ] Predicate-inclusive
- [ ] Add tests for each strategy
- [ ] Document encoding differences

### Phase 3: Variables & Terms (Weeks 5-6)
- [ ] Design and implement `VariableEncoder`
- [ ] Implement `TermEncoder` for structured terms
- [ ] Support nested structures (recursive encoding)
- [ ] Support attribute-value pairs
- [ ] Add grammar support for structured terms

### Phase 4: Rules & Queries (Weeks 7-8)
- [ ] Implement `FormulaEncoder` for rules
- [ ] Implement query encoding options (pattern vs vector)
- [ ] Support conjunctive query encoding
- [ ] Add rule similarity metrics
- [ ] Test with existing chaining system

### Phase 5: Negation & Composites (Weeks 9-10)
- [ ] Implement `NegationEncoder` with multiple strategies
- [ ] Implement `CompositeEncoder` for lists, sets
- [ ] Add temporal encoding support
- [ ] Add spatial encoding support
- [ ] Test integration with existing negation handling

### Phase 6: Meta-Level (Weeks 11-12)
- [ ] Implement context/world encoding
- [ ] Implement certainty/probability encoding
- [ ] Implement provenance encoding
- [ ] Add modality encoding
- [ ] Meta-reasoning capabilities

### Phase 7: User Interface (Weeks 13-14)
- [ ] Implement directive parsing for all encoding configs
- [ ] Create named presets ("standard", "role_filler", "clifford")
- [ ] Add validation and compatibility checking
- [ ] Implement formula-based encoding (advanced)
- [ ] Add encoding introspection tools

### Phase 8: Documentation & Examples (Weeks 15-16)
- [ ] Document all encoding strategies
- [ ] Create example programs for each strategy
- [ ] Add encoding comparison benchmarks
- [ ] Write encoding design guide for advanced users
- [ ] Tutorial on custom encoders

## Design Principles

1. **Modularity**: Each encoding decision is independent and composable
2. **Backward Compatibility**: Default configuration matches current behavior
3. **Transparency**: Users can inspect and understand encoding choices
4. **Extensibility**: Easy to add new encoding strategies
5. **Validation**: Encoding configs validated at parse time, not runtime
6. **Performance**: Encoding overhead minimized via caching and vectorization
7. **Explainability**: Encoding choices reflected in traces and explanations

## Example Configurations

### Standard (Current Default)
```vsarl
@encoding standard {
    symbols: {basis: "random", seed: 42},
    atoms: {strategy: "shift", normalize: true},
    predicates: {include_in_atom: false},
    variables: {representation: "placeholder"},
    rules: {representation: "procedural"},
    queries: {mode: "pattern"},
    negation: {classical: "separate"},
    composite: {lists: "shift", sets: "bundle"}
};
```

### Role-Filler Alternative
```vsarl
@encoding role_filler {
    symbols: {basis: "random", seed: 42},
    atoms: {
        strategy: "role_filler",
        role_type: "random",
        normalize: true
    },
    predicates: {
        include_in_atom: true,
        composition: "bind"
    },
    variables: {representation: "symbol"},
    rules: {representation: "vector"},
    queries: {mode: "vector"},
    composite: {lists: "recursive", sets: "bundle"}
};
```

### Clifford Geometric Algebra Mode
```vsarl
@encoding clifford {
    symbols: {basis: "clifford", grade: 1},
    atoms: {
        strategy: "geometric_product",
        orientation: true
    },
    predicates: {
        include_in_atom: true,
        composition: "geometric_product"
    },
    composite: {
        lists: "trajectory",
        sets: "bundle"
    },
    negation: {classical: "grade_inversion"}
};
```

### Hybrid Experimental
```vsarl
@encoding custom {
    atoms: {
        strategy: "hybrid",
        shift_weight: 0.6,
        role_weight: 0.4
    },
    negation: {
        classical: "phase_inversion",
        naf: "vector"
    },
    meta: {
        certainty: "weight",
        contexts: "bind"
    }
};
```

## Open Questions

1. **Formula-based encoding DSL**: How expressive should it be? Full Python? Restricted DSL?

2. **Encoding versioning**: How to handle KB compatibility when encoding changes?

3. **Performance**: What's the overhead of configurable encoding vs hardcoded?

4. **Defaults**: Should different predicates have different default encodings?

5. **Learning**: Should encodings be learnable from data? (e.g., learned role vectors)

6. **Validation**: How to validate encoding compatibility (e.g., shift requires invertible permutation)?

7. **Optimization**: Can we compile encoding pipelines for performance?

8. **Debugging**: How to visualize and debug encoding decisions?

## Success Criteria

- [ ] Users can specify encoding for all semantic elements
- [ ] At least 3 complete encoding presets available
- [ ] All existing programs work with default encoding
- [ ] New encodings produce measurably different KB behavior
- [ ] Documentation shows encoding experimentation workflow
- [ ] Performance within 20% of hardcoded implementation
- [ ] Encoding choices visible in traces/explanations
- [ ] Clear migration path from v0.4.0

## Next Steps

1. Review and refine this design document
2. Prototype encoder interface hierarchy
3. Implement encoding config dataclasses
4. Refactor current encoding to use new interface
5. Add one alternative encoding (role-filler) as proof of concept
6. Gather user feedback on directive syntax
7. Iterate on design before full implementation

---

**Feedback Welcome**: This is a living document. Please provide feedback on:
- Missing encoding points
- Alternative encoding strategies
- User interface design
- Implementation priorities
