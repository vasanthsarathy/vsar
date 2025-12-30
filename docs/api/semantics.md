# Semantics API Reference

The semantics layer provides the VSAREngine for executing VSAR programs.

## Module: `vsar.semantics.engine`

High-level execution engine orchestrating all VSAR layers.

### VSAREngine

::: vsar.semantics.engine.VSAREngine
    options:
      show_source: true
      heading_level: 4

VSAR execution engine.

Orchestrates all VSAR layers: kernel, symbols, encoding, KB, retrieval, and tracing.

**Example:**

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

# Create engine
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert facts
engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

# Query
result = engine.query(Query(predicate="parent", args=["alice", None]))
```

---

#### `__init__(directives: list[Directive])`

Initialize engine from directives.

**Parameters:**

- `directives` (list[Directive]): Configuration directives

**Raises:**

- `ValueError`: If directives are invalid

**Example:**

```python
from vsar.language.ast import Directive
from vsar.semantics.engine import VSAREngine

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42}),
    Directive(name="threshold", params={"value": 0.22})
]
engine = VSAREngine(directives)
```

**Supported Directives:**

| Directive | Parameters | Default |
|-----------|------------|---------|
| `@model` | `type`, `dim`, `seed` | FHRR, 8192, 42 |
| `@threshold` | `value` | 0.22 |
| `@beam` | `width` | 50 (Phase 2) |

---

#### `insert_fact(fact: Fact)`

Insert a fact into the knowledge base.

**Parameters:**

- `fact` (Fact): Ground fact to insert

**Example:**

```python
from vsar.language.ast import Fact

# Binary fact
engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

# Ternary fact
engine.insert_fact(Fact(predicate="transfer",
                       args=["alice", "bob", "money"]))
```

---

#### `query(query: Query, k: int | None = None) -> QueryResult`

Execute a query with tracing.

**Parameters:**

- `query` (Query): Query to execute
- `k` (int | None): Number of results (default: 10)

**Returns:**

- `QueryResult`: Results with trace ID

**Raises:**

- `ValueError`: If query has != 1 variable
- `ValueError`: If predicate not found in KB

**Example:**

```python
from vsar.language.ast import Query

query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=5)

for entity, score in result.results:
    print(f"{entity}: {score:.4f}")
```

---

#### `stats() -> dict[str, Any]`

Get knowledge base statistics.

**Returns:**

- dict with keys:
  - `total_facts` (int): Total number of facts
  - `predicates` (dict[str, int]): Counts by predicate

**Example:**

```python
stats = engine.stats()
print(f"Total facts: {stats['total_facts']}")

for predicate, count in stats['predicates'].items():
    print(f"  {predicate}: {count} facts")
```

---

#### `export_kb(format: str = "json") -> dict | str`

Export knowledge base.

**Parameters:**

- `format` (str): "json" or "jsonl"

**Returns:**

- dict (json) or str (jsonl)

**Raises:**

- `ValueError`: If format is invalid

**Example:**

```python
# Export as JSON
data = engine.export_kb("json")
# Returns: {"parent": [{"args": ["alice", "bob"]}, ...], ...}

# Export as JSONL
jsonl_data = engine.export_kb("jsonl")
# Returns: '{"predicate": "parent", "args": ["alice", "bob"]}\n...'
```

---

#### `save_kb(path: Path | str)`

Save knowledge base to HDF5 file.

**Parameters:**

- `path` (Path | str): Path to save KB

**Example:**

```python
from pathlib import Path

engine.save_kb("family.h5")
engine.save_kb(Path("data") / "kb.h5")
```

---

#### `load_kb(path: Path | str)`

Load knowledge base from HDF5 file.

**Parameters:**

- `path` (Path | str): Path to load KB from

**Important:** Engine configuration must match the KB's configuration.

**Example:**

```python
from pathlib import Path

engine.load_kb("family.h5")
engine.load_kb(Path("data") / "kb.h5")
```

---

### QueryResult

::: vsar.semantics.engine.QueryResult
    options:
      show_source: true
      heading_level: 4

Result of a query execution.

**Attributes:**

- `query` (Query): Original query
- `results` (list[tuple[str, float]]): List of (entity, score) tuples
- `trace_id` (str): ID of trace event for this query

**Example:**

```python
result = engine.query(query, k=5)

# Access original query
print(f"Query: {result.query.predicate}")

# Access results
for entity, score in result.results:
    print(f"{entity}: {score:.4f}")

# Access trace
trace_events = engine.trace.get_subgraph(result.trace_id)
```

---

## Usage Examples

### Example 1: Basic Usage

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

# Step 1: Create engine
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Step 2: Insert facts
facts = [
    Fact(predicate="parent", args=["alice", "bob"]),
    Fact(predicate="parent", args=["alice", "carol"]),
    Fact(predicate="parent", args=["bob", "dave"]),
]
for fact in facts:
    engine.insert_fact(fact)

# Step 3: Query
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=5)

# Step 4: Process results
for entity, score in result.results:
    print(f"{entity}: {score:.4f}")
```

### Example 2: With File Loaders

```python
from vsar.language.loader import load_csv
from vsar.language.ast import Directive, Query
from vsar.semantics.engine import VSAREngine

# Load facts from CSV
facts = load_csv("people.csv")

# Create engine
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert all facts
for fact in facts:
    engine.insert_fact(fact)

# Multiple queries
queries = [
    Query(predicate="parent", args=["alice", None]),
    Query(predicate="lives_in", args=[None, "boston"]),
]

for query in queries:
    result = engine.query(query, k=10)
    print(f"\nQuery: {query.predicate}")
    for entity, score in result.results:
        print(f"  {entity}: {score:.4f}")
```

### Example 3: Persistence

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

# Create and save KB
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine1 = VSAREngine(directives)

for i in range(1000):
    fact = Fact(predicate="test", args=[f"entity_{i}", f"value_{i}"])
    engine1.insert_fact(fact)

engine1.save_kb("large_kb.h5")
print(f"Saved {engine1.kb.count()} facts")

# Load KB in new session
engine2 = VSAREngine(directives)
engine2.load_kb("large_kb.h5")

# Query loaded KB
query = Query(predicate="test", args=["entity_0", None])
result = engine2.query(query, k=5)
print(f"Loaded KB has {engine2.kb.count()} facts")
```

### Example 4: Different Configurations

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

# High-accuracy configuration
high_acc = [
    Directive(name="model", params={"type": "FHRR", "dim": 16384, "seed": 42}),
    Directive(name="threshold", params={"value": 0.30})
]
engine_hi = VSAREngine(high_acc)

# Fast configuration
fast = [
    Directive(name="model", params={"type": "FHRR", "dim": 1024, "seed": 42}),
    Directive(name="threshold", params={"value": 0.15})
]
engine_fast = VSAREngine(fast)

# Same facts in both
fact = Fact(predicate="parent", args=["alice", "bob"])
engine_hi.insert_fact(fact)
engine_fast.insert_fact(fact)

# Compare results
query = Query(predicate="parent", args=["alice", None])
result_hi = engine_hi.query(query, k=5)
result_fast = engine_fast.query(query, k=5)

print("High-accuracy:", result_hi.results)
print("Fast:", result_fast.results)
```

### Example 5: Export and Analysis

```python
from vsar.language.ast import Directive, Fact
from vsar.semantics.engine import VSAREngine
import json

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert facts
for i in range(100):
    engine.insert_fact(Fact(predicate="test", args=[f"e{i}", f"v{i}"]))

# Get statistics
stats = engine.stats()
print(f"Total facts: {stats['total_facts']}")

# Export as JSON
data = engine.export_kb("json")
with open("export.json", "w") as f:
    json.dump(data, f, indent=2)

# Export as JSONL
jsonl_data = engine.export_kb("jsonl")
with open("export.jsonl", "w") as f:
    f.write(jsonl_data)

print("Exported KB to export.json and export.jsonl")
```

---

## See Also

- [Language API](language.md) - AST and loaders
- [Trace API](trace.md) - Trace layer
- [Python API Guide](../guides/python-api.md) - Usage examples
