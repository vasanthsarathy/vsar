# VSAR Example Programs

This directory contains example VSAR programs demonstrating Phase 2 capabilities (Horn rules + forward chaining).

## Examples

### 01_basic_rules.vsar
**Demonstrates:** Simple rule derivation
- Base facts about persons
- Single rule deriving humans from persons
- Basic query execution

**Key Concepts:**
- Rule syntax: `rule head(X) :- body(X).`
- Single-variable queries

### 02_family_tree.vsar
**Demonstrates:** Multi-hop inference with grandparent rules
- Parent-child relationships
- Grandparent derivation via two-body rule
- Multiple queries (grandchildren, grandparents)

**Key Concepts:**
- Multi-body rules: `rule head(X, Z) :- body1(X, Y), body2(Y, Z).`
- Variable binding across atoms
- Join operations

### 03_transitive_closure.vsar
**Demonstrates:** Recursive rules and transitive closure
- Parent facts across 4 generations
- Ancestor rules (base + recursive cases)
- Multi-generation inference

**Key Concepts:**
- Recursive rules
- Transitive closure (ancestor from parent)
- Forward chaining fixpoint
- Multi-hop derivations

### 04_organizational_hierarchy.vsar
**Demonstrates:** Manager chains and reporting relationships
- Organizational structure (CEO → VPs → Directors → Managers → Employees)
- Reports-to relationship derived from manages
- Transitive reporting chains

**Key Concepts:**
- Real-world hierarchies
- Reverse relationships (manages → reports_to)
- Multi-level transitive inference

### 05_knowledge_graph.vsar
**Demonstrates:** Multiple relation types combined via rules
- Social connections (knows)
- Professional connections (works_with)
- General connection derived from both
- Transitive connection network

**Key Concepts:**
- Multiple predicates
- Rule disjunction (multiple rules with same head)
- Heterogeneous graph reasoning

### 06_academic_network.vsar
**Demonstrates:** Complex multi-rule scenario
- Academic lineage (advisor-student chains)
- Collaboration networks (coauthorship)
- Symmetric relationships
- Multiple interacting rule types

**Key Concepts:**
- Complex rule interactions
- Symmetric rules: `rule R(Y, X) :- R(X, Y).`
- Multiple derivation paths
- Academic sibling detection

## Running Examples

VSAR programs can be executed via the Python API or CLI (when implemented):

### Python API
```python
from vsar.language.loader import ProgramLoader
from vsar.semantics.engine import VSAREngine

# Load program
loader = ProgramLoader()
program = loader.load_file("examples/03_transitive_closure.vsar")

# Create engine from directives
engine = VSAREngine(program.directives)

# Insert facts
for fact in program.facts:
    engine.insert_fact(fact)

# Execute queries
for query in program.queries:
    result = engine.query(query, rules=program.rules, k=10)
    print(f"Query: {query.predicate}({', '.join(str(a) for a in query.args)})")
    print(f"Results: {result.results}")
```

### Manual Execution (Current)
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
        head=Atom(predicate="ancestor", args=["X", "Y"]),
        body=[Atom(predicate="parent", args=["X", "Y"])],
    ),
    Rule(
        head=Atom(predicate="ancestor", args=["X", "Z"]),
        body=[
            Atom(predicate="parent", args=["X", "Y"]),
            Atom(predicate="ancestor", args=["Y", "Z"]),
        ],
    ),
]

# Query with rules
result = engine.query(Query(predicate="ancestor", args=["alice", None]), rules=rules, k=10)
print(result.results)  # [(bob, score), (carol, score)]
```

## What These Examples Demonstrate

### Phase 2 Capabilities
- ✅ Horn clause rules (`head :- body1, body2, ...`)
- ✅ Forward chaining with fixpoint detection
- ✅ Semi-naive evaluation (automatic optimization)
- ✅ Novelty detection (prevents duplicate derivations)
- ✅ Multi-hop inference (arbitrary depth)
- ✅ Transitive closure
- ✅ Beam search joins (controls combinatorial explosion)
- ✅ Query with automatic rule application
- ✅ Full traceability and provenance

### VSA-Based Approximate Reasoning
Unlike traditional Datalog/Prolog systems:
- **Similarity-based retrieval** instead of exact symbolic matching
- **Ranked results** with confidence scores
- **Graceful degradation** under noise (fuzzy matching)
- **Vectorized operations** for performance

### Limitations (Not Yet Implemented)
- ❌ Negation (`not`, `!`)
- ❌ Multi-variable queries (`parent(?, ?)?`)
- ❌ Aggregation (`count`, `sum`, `max`)
- ❌ Backward chaining
- ❌ SPARQL-like path queries

## Next Steps

After Phase 2, future enhancements will add:
- **Phase 3:** Negation and stratified evaluation
- **Phase 4:** Multi-variable queries and aggregation
- **Phase 5:** Backward chaining and magic sets
- **Phase 6:** Advanced optimizations and incremental maintenance

See `PROGRESS.md` for detailed roadmap.
