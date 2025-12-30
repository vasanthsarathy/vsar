# CLI API Reference

The CLI module provides command-line interface for VSAR.

## Module: `vsar.cli.main`

Typer-based CLI application.

### Commands

The VSAR CLI provides four main commands:

- `vsar run` - Execute VSAR program
- `vsar ingest` - Ingest facts from files
- `vsar export` - Export knowledge base
- `vsar inspect` - Show KB statistics

For detailed usage, see [CLI Reference](../cli-reference.md).

---

## Module: `vsar.cli.formatters`

Output formatting utilities.

### format_results_table

Format query results as table.

**Parameters:**

- `results` (list[QueryResult]): Query results

**Returns:**

- str: Formatted table

**Example:**

```python
from vsar.cli.formatters import format_results_table
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
result = engine.query(Query(predicate="parent", args=["alice", None]))

table = format_results_table([result])
print(table)
```

**Output:**

```
┌─────────────────────────┐
│ Query: parent(alice, X) │
├────────┬────────────────┤
│ Entity │ Score          │
├────────┼────────────────┤
│ bob    │ 0.9234         │
└────────┴────────────────┘
```

---

### format_results_json

Format query results as JSON.

**Parameters:**

- `results` (list[QueryResult]): Query results

**Returns:**

- str: JSON string

**Example:**

```python
from vsar.cli.formatters import format_results_json
import json

json_output = format_results_json([result])
data = json.loads(json_output)

print(json.dumps(data, indent=2))
```

**Output:**

```json
{
  "queries": [
    {
      "predicate": "parent",
      "args": ["alice", null],
      "results": [
        {"entity": "bob", "score": 0.9234}
      ]
    }
  ]
}
```

---

### format_trace_dag

Format trace DAG for display.

**Parameters:**

- `trace` (TraceCollector): Trace collector
- `event_id` (str | None): Event ID for subgraph (None = full DAG)

**Returns:**

- str: Formatted trace

**Example:**

```python
from vsar.cli.formatters import format_trace_dag

# Full DAG
trace_output = format_trace_dag(engine.trace, event_id=None)
print(trace_output)

# Subgraph for specific query
trace_output = format_trace_dag(engine.trace, event_id=result.trace_id)
print(trace_output)
```

---

## Usage Examples

### Example 1: Programmatic CLI Usage

```python
import sys
from vsar.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

# Run command
result = runner.invoke(app, ["run", "family.vsar"])
print(result.stdout)

# Ingest command
result = runner.invoke(app, ["ingest", "facts.csv", "--kb", "family.h5"])
print(result.stdout)

# Inspect command
result = runner.invoke(app, ["inspect", "family.h5"])
print(result.stdout)
```

### Example 2: Custom Formatting

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert facts
for i in range(10):
    engine.insert_fact(Fact(predicate="test", args=[f"a{i}", f"b{i}"]))

# Query
result = engine.query(Query(predicate="test", args=["a0", None]), k=5)

# Custom formatting
print("Custom Results:")
print(f"Query: {result.query.predicate}({', '.join(str(a) for a in result.query.args)})")
print(f"Found {len(result.results)} results:\n")

for i, (entity, score) in enumerate(result.results, 1):
    confidence = "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"
    print(f"{i}. {entity:20s} {score:6.4f} ({confidence})")
```

### Example 3: JSON Output for Scripts

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine
from vsar.cli.formatters import format_results_json
import json

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert facts
engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
engine.insert_fact(Fact(predicate="parent", args=["bob", "carol"]))

# Multiple queries
queries = [
    Query(predicate="parent", args=["alice", None]),
    Query(predicate="parent", args=[None, "carol"]),
]

results = [engine.query(q, k=5) for q in queries]

# Format as JSON
json_output = format_results_json(results)
data = json.loads(json_output)

# Process with jq-like operations
for query_result in data["queries"]:
    high_confidence = [
        r for r in query_result["results"]
        if r["score"] > 0.8
    ]
    print(f"High-confidence results for {query_result['predicate']}: {len(high_confidence)}")
```

### Example 4: Trace Visualization

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine
from vsar.cli.formatters import format_trace_dag

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

# Query with trace
result = engine.query(Query(predicate="parent", args=["alice", None]))

# Display trace
trace_output = format_trace_dag(engine.trace, event_id=result.trace_id)
print("Trace for query:")
print(trace_output)

# Full DAG
full_trace = format_trace_dag(engine.trace, event_id=None)
print("\nFull trace DAG:")
print(full_trace)
```

---

## Command Line Examples

For complete CLI usage examples, see [CLI Reference](../cli-reference.md).

### Quick Reference

**Run program:**

```bash
vsar run family.vsar
vsar run family.vsar --json
vsar run family.vsar --trace
vsar run family.vsar --k 10
```

**Ingest facts:**

```bash
vsar ingest facts.csv --kb family.h5
vsar ingest facts.jsonl --kb family.h5
vsar ingest parents.csv --predicate parent --kb family.h5
```

**Export KB:**

```bash
vsar export family.h5 --format json --output facts.json
vsar export family.h5 --format jsonl --output facts.jsonl
```

**Inspect KB:**

```bash
vsar inspect family.h5
```

---

## See Also

- [CLI Reference](../cli-reference.md) - Complete command documentation
- [Semantics API](semantics.md) - VSAREngine
- [Getting Started](../getting-started.md) - CLI tutorials
