# VSAR Complete Rewrite: Test-Driven Implementation Plan

**Date**: 2026-01-01
**Scope**: Complete rewrite of VSAR framework based on new specifications
**Approach**: Test-driven development, incremental phases
**Status**: Planning

---

## Executive Summary

This plan implements a complete rewrite of VSAR based on three specification documents:

1. **VSAR Unified Encoding & Reasoning Framework** - New FHRR-based encoding with bind/unbind
2. **VSAR Inference Engine** - Multi-mode reasoning controllers
3. **VSAR-DSL** - New specification language

### Key Architectural Changes

| Aspect | OLD (v0.4.0) | NEW (Spec) |
|--------|-------------|------------|
| **Encoding** | Shift-based (circular permutation) | Bind-based (⊗/⊘ operations) |
| **Decoding** | Resonator filtering | Structure-informed unbind → typed cleanup |
| **Symbol Spaces** | 6 basic spaces | 10+ typed codebooks with roles/tags/operators |
| **Unification** | Weighted bundle | Slot-level unbind + cleanup |
| **Reasoning Modes** | Horn chaining only | 11 modes (DL, abduction, argumentation, etc.) |
| **Language** | Simple facts/rules/queries | 4-block DSL (signature, semantics, KB, queries) |
| **Data Model** | Vectors + tuples | Items with metadata (weight, priority, agent, provenance) |

### Critical Differences

1. **Use VSAX bind/unbind** (not shift) - spec assumes these work correctly now
2. **Separate cleanup from similarity** - distinct operations with different purposes
3. **Typed codebooks** - strict type system via typed cleanup domains
4. **Multi-mode inference** - pluggable controllers over shared representation
5. **Paraconsistent by default** - track supp(A) and supp(¬A) independently

---

## Phase 0: VSAX Core Integration & Testing

**Goal**: Verify VSAX bind/unbind operations work correctly, establish core VSA primitives

### 0.1 VSAX Operations Validation

**Tests** (`tests/core/test_vsax_primitives.py`):

```python
def test_bind_unbind_identity():
    """Verify: (a ⊗ b) ⊘ b ≈ a"""
    # Test on random vectors
    # Test with different seeds
    # Measure reconstruction error

def test_bind_commutativity():
    """Verify: a ⊗ b == b ⊗ a"""

def test_unbind_inverse():
    """Verify: a ⊘ b == a ⊗ conj(b)"""

def test_bundle_superposition():
    """Verify: (a ⊕ b ⊕ c) contains signals for a, b, c"""
    # Measure similarity to components

def test_cleanup_under_noise():
    """Verify cleanup recovers correct symbol from noisy vector"""
    # Add controlled noise
    # Test cleanup threshold behavior
```

**Implementation**:
- Create thin wrapper around `vsax.FHRRMemory`
- Expose: `bind`, `unbind`, `bundle`, `similarity`, `normalize`
- Module: `src/vsar/kernel/vsax_backend.py`

**Success Criteria**:
- Bind/unbind reconstruction error < 0.01
- Cleanup success rate > 99% for SNR > 0.5
- All primitives pass deterministic tests with fixed seeds

---

## Phase 1: Encoding Layer

**Goal**: Implement FHRR encodings for all logical structures per spec

### 1.1 Symbol Spaces & Typed Codebooks

**Tests** (`tests/encoding/test_symbol_spaces.py`):

```python
def test_create_typed_codebook():
    """Create separate codebooks for E, C, R, F, etc."""

def test_register_symbol_in_space():
    """Register 'alice' in ENTITIES, 'Person' in CONCEPTS"""

def test_symbols_nearly_orthogonal():
    """Verify random basis vectors have low similarity"""

def test_cleanup_typed():
    """Cleanup only searches relevant codebook"""
```

**Implementation** (`src/vsar/symbols/`):
- `spaces.py`: Enum for all symbol spaces (E, C, R, F, P, TAGS, OPS, ROLES, ...)
- `codebook.py`: `TypedCodebook` class wrapping VSAMemory per space
- `registry.py`: `SymbolRegistry` managing all codebooks

**Data Structures**:
```python
class SymbolSpace(Enum):
    ENTITIES = "E"          # Constants/individuals
    CONCEPTS = "C"          # Unary predicates
    ROLES = "R"             # Binary relations
    FUNCTIONS = "F"         # Function symbols
    PREDICATES = "P"        # General predicates
    ARG_ROLES = "ARG"       # ARG₁, ARG₂, ...
    STRUCT_ROLES = "STRUCT" # HEAD, BODY, SRC, TGT, LEFT, RIGHT
    TAGS = "TAG"            # ATOM, TERM, RULE, LIT, META, AXIOM
    OPS = "OP"              # AND, OR, NOT, EXISTS, FORALL
    EPI_OPS = "EPI"         # KNOW, BELIEF
    GRAPH_OPS = "GRAPH"     # EDGE, SUPPORT, ATTACK

class TypedCodebook:
    def __init__(self, space: SymbolSpace, dim: int, seed: int):
        self.space = space
        self.memory = vsax.FHRRMemory(dim=dim, seed=seed)

    def register(self, name: str) -> ndarray:
        """Register symbol and return basis vector"""

    def cleanup(self, vec: ndarray, k: int = 1) -> list[tuple[str, float]]:
        """Find k nearest symbols in this space"""
```

### 1.2 Atom Encoding with Bind

**Tests** (`tests/encoding/test_atom_encoding.py`):

```python
def test_encode_atom_simple():
    """Test: enc(parent(alice, bob))"""
    # Verify structure: (P_parent ⊗ TAG_ATOM) ⊗ (ARG₁ ⊗ E_alice ⊕ ARG₂ ⊗ E_bob)

def test_decode_atom_predicate():
    """Decode predicate from atom encoding"""
    # enc ⊘ TAG_ATOM → should cleanup to 'parent'

def test_decode_atom_arg_positions():
    """Decode specific argument positions"""
    # payload = enc ⊘ (P_parent ⊗ TAG_ATOM)
    # v1 = payload ⊘ ARG₁ → cleanup to 'alice'
    # v2 = payload ⊘ ARG₂ → cleanup to 'bob'

def test_nested_function_terms():
    """Test: enc(parent(alice, mother(bob)))"""
    # Recursive encoding

def test_variable_handling():
    """Variables not in codebook, represented as meta-tokens"""
```

**Implementation** (`src/vsar/encoding/`):
- `atom_encoder.py`: `AtomEncoder` class
- `term_encoder.py`: `TermEncoder` for function terms
- `formula_encoder.py`: Encoders for rules, DL constructs

**Core Encoding Formula**:
```python
class AtomEncoder:
    def encode_atom(self, predicate: str, args: list[Term]) -> ndarray:
        """
        enc(p(t1,...,tk)) = (P_p ⊗ TAG_ATOM) ⊗ (⊕ᵢ ARGᵢ ⊗ enc(tᵢ))
        """
        # 1. Get predicate vector
        pred_vec = self.registry.get(SymbolSpace.PREDICATES, predicate)
        tag_atom = self.registry.get(SymbolSpace.TAGS, "ATOM")

        # 2. Encode arguments
        arg_bundles = []
        for i, arg in enumerate(args):
            arg_role = self.registry.get(SymbolSpace.ARG_ROLES, f"ARG{i+1}")
            arg_vec = self.encode_term(arg)
            arg_bundles.append(self.backend.bind(arg_role, arg_vec))

        # 3. Bundle arguments
        args_bundle = self.backend.bundle(arg_bundles)

        # 4. Bind predicate+tag with args
        head = self.backend.bind(pred_vec, tag_atom)
        result = self.backend.bind(head, args_bundle)

        return self.backend.normalize(result)
```

### 1.3 DL Constructor Encoding

**Tests** (`tests/encoding/test_dl_encoding.py`):

```python
def test_encode_concept_conjunction():
    """Test: enc(Person ⊓ Doctor)"""

def test_encode_concept_negation():
    """Test: enc(¬Person)"""

def test_encode_existential_restriction():
    """Test: enc(∃hasChild.Doctor)"""

def test_decode_dl_constructor():
    """Decode LEFT, RIGHT, ROLE, FILLER slots"""

def test_encode_tbox_axiom():
    """Test: enc(Doctor ⊑ Person)"""

def test_encode_abox_assertion():
    """Test: enc(Doctor(alice))"""
```

### 1.4 Rule & Meta-Level Encoding

**Tests** (`tests/encoding/test_rule_encoding.py`):

```python
def test_encode_horn_rule():
    """Test: enc(rule H :- B1, B2)"""
    # TAG_RULE ⊗ (HEAD ⊗ enc(H) ⊕ BODY ⊗ enc(B1 ∧ B2))

def test_encode_negation_types():
    """Test classical (~A) vs NAF (not A)"""

def test_encode_epistemic():
    """Test: enc(K_a φ)"""

def test_encode_argument_edge():
    """Test: enc(support(A, B))"""
```

**Success Criteria**:
- All encoding tests pass
- Decode-then-cleanup recovers original symbols
- Nested structures encode/decode correctly

---

## Phase 2: Unification Kernel

**Goal**: Structure-aware decoding with unbind → typed cleanup

### 2.1 Slot-Level Decoding

**Tests** (`tests/unification/test_decoding.py`):

```python
def test_decode_atom_slots():
    """Given enc(parent(alice,bob)), decode each slot"""
    # 1. Unbind predicate+tag
    # 2. Unbind each argument role
    # 3. Cleanup to symbols

def test_decode_with_variables():
    """Handle unbound variable positions"""

def test_decode_nested_terms():
    """Recursively decode function terms"""

def test_cleanup_threshold():
    """Abstain if similarity < threshold"""
```

**Implementation** (`src/vsar/unification/`):
- `decoder.py`: `StructureDecoder` class
- `substitution.py`: `Substitution` (variable bindings)
- `unifier.py`: `Unifier` (pattern matching)

**Decoding Algorithm**:
```python
class StructureDecoder:
    def decode_atom(self, vec: ndarray, threshold: float = 0.25) -> Atom | None:
        """
        Decode atom vector via unbind → cleanup

        Returns None if any cleanup below threshold
        """
        # 1. Extract predicate
        tag_atom = self.registry.get(SymbolSpace.TAGS, "ATOM")
        payload = self.backend.unbind(vec, tag_atom)

        pred_candidates = self.registry.cleanup(SymbolSpace.PREDICATES, payload, k=1)
        if pred_candidates[0][1] < threshold:
            return None  # UNKNOWN

        predicate = pred_candidates[0][0]

        # 2. Get arity
        arity = self.signature.get_arity(predicate)

        # 3. Unbind predicate to get args bundle
        pred_vec = self.registry.get(SymbolSpace.PREDICATES, predicate)
        args_bundle = self.backend.unbind(payload, pred_vec)

        # 4. Decode each argument
        args = []
        for i in range(arity):
            arg_role = self.registry.get(SymbolSpace.ARG_ROLES, f"ARG{i+1}")
            arg_vec = self.backend.unbind(args_bundle, arg_role)

            # Try constants first
            const_candidates = self.registry.cleanup(SymbolSpace.ENTITIES, arg_vec, k=1)
            if const_candidates[0][1] >= threshold:
                args.append(Constant(const_candidates[0][0]))
            else:
                # Try terms/functions
                # Or return Variable token
                args.append(Variable(f"X{i+1}"))

        return Atom(predicate, args)
```

### 2.2 Unification via Decoding

**Tests** (`tests/unification/test_unification.py`):

```python
def test_unify_ground_atoms():
    """unify(parent(alice,bob), parent(alice,bob)) → {}"""

def test_unify_with_variable():
    """unify(parent(alice,X), parent(alice,bob)) → {X: bob}"""

def test_unify_failure():
    """unify(parent(alice,bob), parent(carol,bob)) → None"""

def test_unify_nested_terms():
    """unify(parent(X, f(Y)), parent(alice, f(bob)))"""

def test_substitution_composition():
    """Compose multiple substitutions"""
```

**Success Criteria**:
- Decode accuracy > 95% on clean encodings
- Decode handles controlled noise gracefully
- Unification produces correct substitutions

---

## Phase 3: Core Data Stores

**Goal**: Item schema, fact/rule stores, paraconsistent tracking

### 3.1 Item Schema & Metadata

**Tests** (`tests/store/test_item_schema.py`):

```python
def test_create_fact_item():
    """Create Item with kind=FACT, weight, provenance"""

def test_create_rule_item():
    """Create Item with kind=RULE, priority, exceptions"""

def test_paraconsistent_belief_record():
    """Maintain (supp_pos, supp_neg) for literal"""

def test_update_belief_state():
    """Add support for L and ~L independently"""
```

**Implementation** (`src/vsar/store/`):
- `item.py`: `Item` dataclass
- `belief.py`: `BeliefState` (paraconsistent tracking)
- `fact_store.py`: `FactStore` class
- `rule_store.py`: `RuleStore` class

**Data Structures**:
```python
@dataclass
class Item:
    vec: ndarray              # FHRR encoding
    kind: ItemKind            # FACT, RULE, AXIOM, EDGE, CASE, MAP
    weight: float             # Probability/confidence
    priority: float | None    # For defaults
    agent: str | None         # For epistemic
    provenance: Provenance    # Source, timestamp, trace
    tags: set[str]

class ItemKind(Enum):
    FACT = "fact"
    RULE = "rule"
    AXIOM = "axiom"
    EDGE = "edge"
    CASE = "case"
    MAP = "map"

@dataclass
class Provenance:
    source: str
    timestamp: datetime
    agent: str | None
    trace: list[str]  # Derivation trace

@dataclass
class BeliefState:
    """Paraconsistent belief record for literal L"""
    supp_pos: float   # Support for L
    supp_neg: float   # Support for ¬L

    def is_consistent(self) -> bool:
        return self.supp_pos > 0 and self.supp_neg == 0

    def is_contradictory(self) -> bool:
        return self.supp_pos > 0 and self.supp_neg > 0

    def is_unknown(self) -> bool:
        return self.supp_pos == 0 and self.supp_neg == 0
```

### 3.2 Predicate-Indexed Stores

**Tests** (`tests/store/test_indexing.py`):

```python
def test_index_facts_by_predicate():
    """Retrieve facts with predicate 'parent'"""

def test_retrieve_candidates():
    """Retrieve top-k similar facts"""

def test_check_novelty():
    """Check if similar fact already exists"""

def test_agent_indexed_kb():
    """Separate KB per agent for epistemic"""
```

**Implementation**:
```python
class FactStore:
    def __init__(self):
        self._by_predicate: dict[str, list[Item]] = {}
        self._belief_state: dict[str, BeliefState] = {}

    def insert(self, item: Item, literal: Literal):
        """Insert fact and update belief state"""
        # Add to predicate index
        # Update belief state (supp_pos or supp_neg)

    def retrieve_by_predicate(self, predicate: str) -> list[Item]:
        """Get all facts with predicate"""

    def retrieve_similar(self, query_vec: ndarray, k: int) -> list[Item]:
        """Exploratory similarity search"""

    def get_belief(self, literal: Literal) -> BeliefState:
        """Get paraconsistent belief state"""
```

**Success Criteria**:
- Items store all metadata correctly
- Paraconsistent tracking maintains independent supports
- Indexing enables O(1) predicate lookup

---

## Phase 4: Minimal Viable Reasoning (Horn + Cleanup)

**Goal**: Implement core deductive reasoning to validate architecture

### 4.1 Query Answering with Unbind→Cleanup

**Tests** (`tests/reasoning/test_query.py`):

```python
def test_crisp_query_answering():
    """
    Facts: parent(alice, bob), parent(alice, carol)
    Query: parent(alice, ?X)
    Expected: [bob, carol] with high scores
    """

def test_query_with_bound_args():
    """Query: parent(?X, bob) → alice"""

def test_query_no_matches():
    """Query: parent(dave, ?X) → []"""

def test_query_below_threshold():
    """Noisy encoding → UNKNOWN result"""
```

**Implementation** (`src/vsar/reasoning/`):
- `query_engine.py`: `QueryEngine` class
- `retriever.py`: Fact retrieval

**Query Algorithm** (from spec):
```python
class QueryEngine:
    def answer_query(self, query: Atom) -> list[tuple[str, float]]:
        """
        Answer query p(a, ?X) via:
        1. Retrieve candidate facts with predicate p
        2. For each fact, decode slots and verify bound args
        3. Decode unknown slot via unbind
        4. Cleanup to typed symbol
        5. Return all cleaned answers above threshold
        """
        # 1. Get facts by predicate index
        candidates = self.fact_store.retrieve_by_predicate(query.predicate)

        results = []
        for fact_item in candidates:
            # 2. Decode fact
            decoded = self.decoder.decode_atom(fact_item.vec)
            if decoded is None:
                continue

            # 3. Check bound arguments match
            if not self._match_bound_args(decoded, query):
                continue

            # 4. Extract unbound argument
            var_pos = self._get_variable_position(query)
            answer = decoded.args[var_pos]

            # 5. Compute score
            score = fact_item.weight
            results.append((answer, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
```

### 4.2 Forward Chaining (Semi-Naive)

**Tests** (`tests/reasoning/test_chaining.py`):

```python
def test_forward_chaining_simple():
    """
    Facts: parent(alice, bob), parent(bob, carol)
    Rule: grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
    Derive: grandparent(alice, carol)
    """

def test_semi_naive_optimization():
    """Verify only applies rule when body predicates have new facts"""

def test_fixpoint_detection():
    """Stop when no new facts derived"""

def test_weighted_rule_application():
    """Combine weights via t-norm (product)"""
```

**Implementation**:
```python
class ForwardChainer:
    def apply_rules(
        self,
        rules: list[Item],
        max_iterations: int = 100,
        semi_naive: bool = True
    ) -> ChainingResult:
        """
        Semi-naive forward chaining:
        - Track which predicates have new facts (delta)
        - Only apply rule if body predicates in delta
        - Iterate until fixpoint
        """
```

### 4.3 Integration Test

**Test** (`tests/integration/test_mvp.py`):

```python
def test_end_to_end_reasoning():
    """
    Complete flow:
    1. Register symbols in typed codebooks
    2. Encode facts with bind
    3. Encode rule
    4. Run forward chaining
    5. Query derived facts
    6. Verify decoded results
    """
```

**Success Criteria**:
- Query answering returns crisp results via unbind→cleanup
- Forward chaining derives correct facts
- Semi-naive optimization reduces iterations
- End-to-end test passes with >90% accuracy

---

## Phase 5: Description Logic Reasoning

**Goal**: Implement ALC tableau for DL reasoning

### 5.1 DL Constructs

**Tests** (`tests/reasoning/test_dl.py`):

```python
def test_concept_subsumption():
    """TBox: Doctor ⊑ Person → entails Doctor(alice) → Person(alice)"""

def test_concept_conjunction():
    """Test: (Person ⊓ Doctor)(alice)"""

def test_existential_restriction():
    """Test: (∃hasChild.Doctor)(alice)"""

def test_tableau_expansion():
    """Test ⊓-rule, ⊔-rule, ∃-rule, ∀-rule"""

def test_clash_detection():
    """Test: C(x) and ¬C(x) → clash"""
```

**Implementation** (`src/vsar/reasoning/dl/`):
- `tableau.py`: `TableauReasoner` class
- `completion.py`: Completion graph
- `expansion_rules.py`: ALC expansion rules

### 5.2 Paraconsistent DL

**Test**:
```python
def test_paraconsistent_dl():
    """
    If paraconsistent mode enabled:
    - Clash does not terminate tableau
    - Updates belief state instead
    - Returns (supp_pos, supp_neg)
    """
```

**Success Criteria**:
- ALC tableau matches reference reasoner on standard tests
- Paraconsistent mode handles contradictions gracefully

---

## Phase 6-13: Extended Reasoning Modes

**Implementation Order** (from spec):

### Phase 6: Paraconsistent Inference
- 4-valued belief updates
- Query returns (supp_pos, supp_neg)
- Tests: non-explosion under A ∧ ¬A

### Phase 7: Default Reasoning
- Defeasible rules with priorities
- Defeat relation (priority, weight, specificity)
- Warrant computation
- Tests: tweety example (birds fly, penguins don't)

### Phase 8: Probabilistic Inference
- Weighted propagation
- T-norms (product, min, Lukasiewicz)
- T-conorms (noisy-OR, max, logsumexp)
- Tests: probability aggregation

### Phase 9: Argumentation
- AF/BAF graph construction
- Support/attack edges
- Gradual acceptability (iterative strength update)
- Tests: preferred extensions, gradual semantics

### Phase 10: Epistemic Reasoning
- Agent-indexed KBs
- K/B operators (knowledge/belief)
- Nested modalities
- Tests: agent separation, belief updates

### Phase 11: Abduction
- Goal-driven hypothesis generation
- Explanation scoring (coverage, simplicity, consistency)
- Tests: recover known hidden causes

### Phase 12: Analogical Reasoning
- Structure mapping (maximize preserved relations)
- Constraint satisfaction
- Transfer
- Tests: relational structure preservation

### Phase 13: Case-Based Reasoning
- kNN retrieval by similarity
- Adaptation via analogical mapping
- Case storage
- Tests: retrieval/adaptation correctness

### Phase 14: Induction
- Anti-unification
- Rule template generation
- Scoring (coverage, simplicity)
- Tests: learn rules from examples

**Each phase includes**:
- Controller class
- Test suite
- Integration with orchestration loop
- Documentation

---

## Phase 15: VSAR-DSL

**Goal**: New specification language with 4-block structure

### 15.1 Grammar & Parser

**Tests** (`tests/language/test_parser.py`):

```python
def test_parse_signature_block():
    """Parse type/pred/func/role/agent declarations"""

def test_parse_semantics_block():
    """Parse controller selection and policies"""

def test_parse_kb_block():
    """Parse facts, rules, defaults, DL axioms, edges, cases"""

def test_parse_queries_block():
    """Parse entails, explain, analogize, retrieve_case, etc."""

def test_parse_complete_program():
    """Parse all 4 blocks together"""
```

**Implementation** (`src/vsar/language/`):
- `grammar.lark`: Lark grammar for VSAR-DSL
- `parser.py`: Parser using Lark
- `ast.py`: AST node classes

**Grammar Structure** (from spec):
```
program := signature semantics? kb queries?

signature := "SIGNATURE" "{" decl* "}"
decl := type_decl | pred_decl | func_decl | role_decl | agent_decl

semantics := "SEMANTICS" "{" setting* "}"
setting := IDENT ":" value ";"

kb := "KB" "{" stmt* "}"
stmt := fact | rule | default_rule | dl_axiom | arg_edge | case_stmt

queries := "QUERIES" "{" query* "}"
query := "ask" ask_expr meta? "."
ask_expr := lit | "entails" "(" lit ")" | "explain" "(" lit ")" | ...
```

### 15.2 Type System

**Tests** (`tests/language/test_types.py`):

```python
def test_type_checking():
    """Verify argument types match predicate signature"""

def test_typed_cleanup_domains():
    """Ensure cleanup uses correct codebook based on type"""
```

**Implementation** (`src/vsar/language/types.py`):
```python
class TypeChecker:
    def check_program(self, program: Program) -> list[TypeError]:
        """Validate all types in program"""

    def infer_cleanup_space(self, var_type: Type) -> SymbolSpace:
        """Map type to cleanup codebook"""
```

### 15.3 Compiler

**Tests** (`tests/language/test_compiler.py`):

```python
def test_compile_facts_to_items():
    """Compile KB facts to Item objects with encodings"""

def test_compile_rules_to_items():
    """Compile rules to encoded rule items"""

def test_compile_semantics_to_config():
    """Extract controller configuration from SEMANTICS block"""

def test_compile_queries_to_operations():
    """Compile queries to engine API calls"""
```

**Implementation** (`src/vsar/language/compiler.py`):
```python
class VSARCompiler:
    def compile(self, program: Program) -> CompiledProgram:
        """
        Compile VSAR-DSL program to:
        1. Typed codebooks (from SIGNATURE)
        2. Encoded items (from KB)
        3. Controller config (from SEMANTICS)
        4. Query operations (from QUERIES)
        """
```

**Success Criteria**:
- Parser handles all DSL constructs from spec
- Type checker catches errors
- Compiler produces valid engine API calls
- Round-trip: parse → compile → execute → correct results

---

## Phase 16: Orchestration Engine

**Goal**: Unified engine exercising all reasoning modes

### 16.1 Controller Portfolio

**Tests** (`tests/engine/test_orchestration.py`):

```python
def test_orchestration_loop():
    """
    On update:
    1. Update belief state (paraconsistent)
    2. Run deductive closure (Horn)
    3. Run DL completion
    4. Recompute argument graph
    5. Apply defaults subject to defeat
    """

def test_query_routing():
    """
    Route query to correct controller:
    - entails() → deductive/DL
    - explain() → abduction
    - analogize() → mapping
    - retrieve_case() → CBR
    """
```

**Implementation** (`src/vsar/engine/`):
- `orchestrator.py`: `VSAREngine` main class
- `controller.py`: Base `Controller` protocol
- Controllers in `src/vsar/reasoning/*/controller.py`

**Orchestration Loop** (from spec):
```python
class VSAREngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.codebooks = create_codebooks(config)
        self.encoder = AtomEncoder(self.codebooks)
        self.decoder = StructureDecoder(self.codebooks)
        self.stores = {
            'facts': FactStore(),
            'rules': RuleStore(),
            'graph': GraphStore(),
            'cases': CaseStore(),
        }
        self.controllers = self._init_controllers(config)

    def update(self, items: list[Item]):
        """
        On each update:
        1. Update belief state (paraconsistent)
        2. Run deductive closure (Horn)
        3. Run DL completion
        4. Recompute argument graph
        5. Apply defaults subject to defeat
        """
        # Update stores
        for item in items:
            self._update_store(item)

        # Run active controllers
        if 'horn' in self.controllers:
            self.controllers['horn'].update()
        if 'dl' in self.controllers:
            self.controllers['dl'].update()
        if 'argumentation' in self.controllers:
            self.controllers['argumentation'].update()
        if 'defaults' in self.controllers:
            self.controllers['defaults'].update()

    def query(self, query: Query) -> QueryResult:
        """Route query to appropriate controller"""
        if query.type == 'entails':
            return self.controllers['horn'].query(query)
        elif query.type == 'explain':
            return self.controllers['abduction'].query(query)
        elif query.type == 'analogize':
            return self.controllers['analogy'].query(query)
        # ... etc
```

### 16.2 Integration Tests

**Test** (`tests/integration/test_full_system.py`):

```python
def test_tweety_example():
    """
    Complete example from spec exercising:
    - Facts
    - Horn rules
    - Defaults with exceptions
    - Paraconsistent tracking
    - Argumentation
    - Query with belief state
    """
```

**Success Criteria**:
- All controllers integrate smoothly
- Orchestration loop handles updates correctly
- Query routing works for all query types
- Full system tests pass

---

## Phase 17: CLI & REPL

**Goal**: Port existing CLI/IDE to new system

### 17.1 CLI Commands

**Tests** (`tests/cli/test_commands.py`):

```python
def test_cli_run_program():
    """vsar run program.vsar"""

def test_cli_repl():
    """vsar repl - interactive mode"""

def test_cli_inspect():
    """vsar inspect <kb|trace|beliefs>"""
```

**Implementation** (`src/vsar/cli/`):
- Port existing Typer-based CLI
- Adapt to new compiler/engine

### 17.2 IDE

**Tests** (`tests/ide/test_ide_integration.py`):

```python
def test_ide_syntax_highlighting():
    """Highlight new DSL syntax"""

def test_ide_run_program():
    """Execute program in IDE"""

def test_ide_inspect_results():
    """View decoded results, traces, beliefs"""
```

**Implementation** (`src/vsar/ide/`):
- Port existing Tkinter IDE
- Update syntax highlighter for new DSL
- Add belief state viewer (paraconsistent)
- Add trace viewer

**Success Criteria**:
- CLI works with new DSL
- REPL allows incremental updates
- IDE provides rich debugging

---

## Testing Strategy

### Test Pyramid

```
        Integration Tests (10%)
       /                      \
      /   Controller Tests      \
     /         (20%)              \
    /                              \
   /    Unit Tests (Encoding,       \
  /     Decoding, Stores - 70%)      \
 /________________________________________\
```

### Core Test Suites

1. **Primitive Tests** (`tests/core/`): VSAX operations
2. **Encoding Tests** (`tests/encoding/`): All encoding formulas
3. **Decoding Tests** (`tests/unification/`): Unbind→cleanup
4. **Store Tests** (`tests/store/`): Items, beliefs, indexing
5. **Reasoning Tests** (`tests/reasoning/`): Each controller
6. **Language Tests** (`tests/language/`): Parser, compiler
7. **Integration Tests** (`tests/integration/`): End-to-end

### Test Coverage Target

- **Core primitives**: 100%
- **Encoding/decoding**: 100%
- **Stores**: 95%
- **Reasoning controllers**: 90%
- **Language**: 95%
- **Overall**: 90%+

### Benchmark Tests

From spec Section 15:

```python
def test_bind_unbind_identity():
    """Bind/unbind reconstruction error < 0.01"""

def test_cleanup_under_noise():
    """Cleanup success > 99% for SNR > 0.5"""

def test_unification_correctness():
    """Unification matches symbolic baseline"""

def test_horn_closure_correctness():
    """Horn closure matches symbolic reasoner on toy KB"""

def test_alc_tableau_correctness():
    """ALC tableau matches reference reasoner"""

def test_paraconsistent_non_explosion():
    """Derive A ∧ ¬A does not trivialize"""

def test_default_defeat():
    """Defeat behaves by priority"""

def test_abduction_recovery():
    """Abduction recovers known hidden causes"""

def test_argumentation_semantics():
    """Acceptability aligns with chosen semantics"""

def test_analogy_preservation():
    """Analogy transfers preserve relational structure"""

def test_cbr_correctness():
    """CBR retrieval/adaptation correctness"""
```

---

## Migration Strategy

### Rewrite vs. Evolve

**Decision**: Complete rewrite in new codebase

**Reasons**:
1. Encoding changes are fundamental (shift → bind)
2. Data model is completely different (Item schema)
3. Multi-mode architecture vs. single-mode
4. New language with 4-block structure

**Approach**:
1. Create new repo/branch: `vsar-v2`
2. Implement phases incrementally
3. Keep v0.4.0 stable for comparison
4. Port examples to new DSL
5. Benchmark against v0.4.0 on Horn reasoning
6. Release v1.0.0 when feature-complete

### Compatibility

- **Breaking**: Complete API rewrite
- **Migration tool**: Provide `vsar-v1-to-v2` converter for simple programs
- **Documentation**: Migration guide

---

## Implementation Order Summary

| Phase | Component | Duration | Dependencies |
|-------|-----------|----------|-------------|
| 0 | VSAX Integration | 1 week | None |
| 1 | Encoding Layer | 2 weeks | Phase 0 |
| 2 | Unification Kernel | 2 weeks | Phase 1 |
| 3 | Core Stores | 1 week | Phase 2 |
| 4 | MVP (Horn + Query) | 2 weeks | Phase 3 |
| 5 | Description Logic | 2 weeks | Phase 4 |
| 6 | Paraconsistent | 1 week | Phase 4 |
| 7 | Defaults | 1 week | Phase 6 |
| 8 | Probabilistic | 1 week | Phase 4 |
| 9 | Argumentation | 2 weeks | Phase 6 |
| 10 | Epistemic | 1 week | Phase 4 |
| 11 | Abduction | 1 week | Phase 4 |
| 12 | Analogical | 2 weeks | Phase 4 |
| 13 | Case-Based | 1 week | Phase 12 |
| 14 | Induction | 2 weeks | Phase 4 |
| 15 | VSAR-DSL | 3 weeks | All reasoning |
| 16 | Orchestration | 2 weeks | All phases |
| 17 | CLI & IDE | 2 weeks | Phase 15-16 |

**Total**: ~30 weeks (~7 months)

---

## Risk Mitigation

### Risk 1: VSAX bind/unbind still broken

**Mitigation**:
- Phase 0 tests validate this immediately
- If broken, need to fix in VSAX first or find workaround
- Fallback: hybrid encoding (bind for some, shift for others)

### Risk 2: Cleanup accuracy insufficient

**Mitigation**:
- Dimensionality tuning (test 8192, 16384, 32768)
- Threshold adjustment
- Ensemble cleanup (vote across multiple probes)

### Risk 3: Multi-mode complexity

**Mitigation**:
- MVP-first approach (Phase 4)
- Incremental controller addition
- Each controller independently tested
- Clear interfaces between components

### Risk 4: Performance

**Mitigation**:
- JAX vectorization throughout
- Batch operations
- Predicate indexing
- Lazy computation
- Profile and optimize hot paths

---

## Success Criteria (Overall)

1. **Correctness**:
   - All primitive tests pass (bind/unbind identity)
   - Encoding round-trip accuracy > 95%
   - Horn reasoning matches symbolic baseline
   - Each reasoning mode has validated tests

2. **Completeness**:
   - All 11 reasoning modes implemented
   - VSAR-DSL supports all spec constructs
   - CLI and IDE functional

3. **Performance**:
   - Query latency < 100ms for 1000 facts
   - Forward chaining handles 10K facts
   - Scales to 100K symbols in codebooks

4. **Usability**:
   - Example programs for each reasoning mode
   - Documentation covers all features
   - Migration guide from v0.4.0

---

## Next Steps

1. **Review this plan with user** - clarify questions, adjust priorities
2. **Set up project structure** - create vsar-v2 repo, configure testing
3. **Begin Phase 0** - validate VSAX bind/unbind operations
4. **Iterate incrementally** - test-driven development throughout

---

**END OF PLAN**
