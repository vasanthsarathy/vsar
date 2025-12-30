# Python API Guide

This guide covers using VSAR programmatically from Python applications.

## Table of Contents

1. [Installation](#installation)
2. [High-Level API](#high-level-api)
3. [Loading Data](#loading-data)
4. [Executing Queries](#executing-queries)
5. [Working with Results](#working-with-results)
6. [Trace Analysis](#trace-analysis)
7. [Advanced Usage](#advanced-usage)

---

## Installation

Install VSAR via pip:

```bash
pip install vsar
```

Import the main components:

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine
from vsar.language.loader import load_csv, load_jsonl, load_vsar
```

---

## High-Level API

The `VSAREngine` class is the main entry point for programmatic use.

### Basic Example

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

# Step 1: Configure engine
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

# Step 3: Execute query
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=5)

# Step 4: Process results
print(f"Query: parent(alice, X)")
for entity, score in result.results:
    print(f"  {entity}: {score:.4f}")
```

**Output:**

```
Query: parent(alice, X)
  bob: 0.9234
  carol: 0.9156
```

### VSAREngine API

**`__init__(directives: list[Directive])`**

Create engine with configuration.

```python
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42}),
    Directive(name="threshold", params={"value": 0.22})
]
engine = VSAREngine(directives)
```

**`insert_fact(fact: Fact)`**

Add a fact to the knowledge base.

```python
fact = Fact(predicate="lives_in", args=["alice", "boston"])
engine.insert_fact(fact)
```

**`query(query: Query, k: int | None = None) -> QueryResult`**

Execute a query.

```python
query = Query(predicate="lives_in", args=[None, "boston"])
result = engine.query(query, k=10)
```

**`stats() -> dict`**

Get KB statistics.

```python
stats = engine.stats()
print(f"Total facts: {stats['total_facts']}")
print("Predicates:", stats['predicates'])
```

**`save_kb(path: Path | str)`**

Save KB to HDF5 file.

```python
engine.save_kb("family.h5")
```

**`load_kb(path: Path | str)`**

Load KB from HDF5 file.

```python
engine.load_kb("family.h5")
```

**`export_kb(format: str = "json") -> dict | str`**

Export KB to JSON or JSONL.

```python
data = engine.export_kb("json")
jsonl = engine.export_kb("jsonl")
```

---

## Loading Data

VSAR provides loaders for CSV, JSONL, and VSAR files.

### Load from CSV

```python
from vsar.language.loader import load_csv

# CSV with predicate column
facts = load_csv("facts.csv")

# CSV without predicate column
facts = load_csv("parents.csv", predicate="parent")

# Insert into engine
for fact in facts:
    engine.insert_fact(fact)
```

### Load from JSONL

```python
from vsar.language.loader import load_jsonl

facts = load_jsonl("facts.jsonl")

for fact in facts:
    engine.insert_fact(fact)
```

### Load from VSAR File

```python
from vsar.language.loader import load_vsar

program = load_vsar("family.vsar")

# Create engine from directives
engine = VSAREngine(program.directives)

# Insert facts
for fact in program.facts:
    engine.insert_fact(fact)

# Execute queries
for query in program.queries:
    result = engine.query(query)
    print(f"Query: {query.predicate}({', '.join(str(a) for a in query.args)})")
    for entity, score in result.results:
        print(f"  {entity}: {score:.4f}")
```

### Auto-Detect Format

```python
from vsar.language.loader import load_facts

# Auto-detects from extension
facts = load_facts("data.csv", format="auto")
facts = load_facts("data.jsonl", format="auto")
```

---

## Executing Queries

### Basic Queries

```python
# Find children of alice
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=5)

# Find parents of bob
query = Query(predicate="parent", args=[None, "bob"])
result = engine.query(query, k=5)

# Who lives in boston
query = Query(predicate="lives_in", args=[None, "boston"])
result = engine.query(query, k=10)
```

### Query with Different k Values

```python
query = Query(predicate="parent", args=["alice", None])

# Top 5 results
result = engine.query(query, k=5)

# Top 10 results
result = engine.query(query, k=10)

# Top 100 results
result = engine.query(query, k=100)
```

### Ternary Queries

```python
# Three-argument predicates
facts = [
    Fact(predicate="transfer", args=["alice", "bob", "money"]),
    Fact(predicate="transfer", args=["alice", "carol", "book"]),
]

for fact in facts:
    engine.insert_fact(fact)

# Query each position
q1 = Query(predicate="transfer", args=["alice", None, "money"])
q2 = Query(predicate="transfer", args=[None, "bob", "money"])
q3 = Query(predicate="transfer", args=["alice", "bob", None])

for query in [q1, q2, q3]:
    result = engine.query(query)
    print(f"Results: {result.results}")
```

---

## Working with Results

### QueryResult Structure

```python
result = engine.query(query, k=5)

# Access query
print(result.query.predicate)
print(result.query.args)

# Access results (list of tuples: (entity, score))
for entity, score in result.results:
    print(f"{entity}: {score:.4f}")

# Access trace ID
print(f"Trace ID: {result.trace_id}")
```

### Filtering Results

```python
result = engine.query(query, k=100)

# High-confidence only (score > 0.8)
high_conf = [(e, s) for e, s in result.results if s > 0.8]

# Top 5 by score
top5 = sorted(result.results, key=lambda x: x[1], reverse=True)[:5]

# Above threshold
threshold = 0.5
above_threshold = [(e, s) for e, s in result.results if s >= threshold]
```

### Extracting Entities

```python
result = engine.query(query, k=10)

# Get just entity names
entities = [entity for entity, _ in result.results]

# Get entities with high confidence
high_conf_entities = [e for e, s in result.results if s > 0.8]
```

### Result Statistics

```python
result = engine.query(query, k=100)

# Count results
num_results = len(result.results)

# Average score
if result.results:
    avg_score = sum(s for _, s in result.results) / len(result.results)
    print(f"Average score: {avg_score:.4f}")

# Max/min scores
if result.results:
    scores = [s for _, s in result.results]
    print(f"Max score: {max(scores):.4f}")
    print(f"Min score: {min(scores):.4f}")
```

---

## Trace Analysis

Use the trace layer to understand query execution.

### Accessing Trace

```python
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query)

# Get trace for this query
trace_events = engine.trace.get_subgraph(result.trace_id)

for event in trace_events:
    print(f"{event.type}: {event.payload}")
```

### Complete Trace DAG

```python
# Execute multiple queries
query1 = Query(predicate="parent", args=["alice", None])
query2 = Query(predicate="parent", args=[None, "bob"])

result1 = engine.query(query1)
result2 = engine.query(query2)

# Get full trace DAG
full_trace = engine.trace.get_dag()

print(f"Total trace events: {len(full_trace)}")

# Analyze by type
query_events = [e for e in full_trace if e.type == "query"]
retrieval_events = [e for e in full_trace if e.type == "retrieval"]

print(f"Query events: {len(query_events)}")
print(f"Retrieval events: {len(retrieval_events)}")
```

### Export Trace

```python
# Get trace as dictionary
trace_dict = engine.trace.to_dict()

import json
with open("trace.json", "w") as f:
    json.dump(trace_dict, f, indent=2)
```

---

## Advanced Usage

### Custom Configuration

```python
# High-accuracy configuration
directives_hi_acc = [
    Directive(name="model", params={"type": "FHRR", "dim": 16384, "seed": 42}),
    Directive(name="threshold", params={"value": 0.30})
]
engine_hi_acc = VSAREngine(directives_hi_acc)

# Fast configuration
directives_fast = [
    Directive(name="model", params={"type": "FHRR", "dim": 1024, "seed": 42}),
    Directive(name="threshold", params={"value": 0.15})
]
engine_fast = VSAREngine(directives_fast)
```

### Batch Processing

```python
import csv

# Process large CSV in batches
batch_size = 1000
with open("large.csv") as f:
    reader = csv.reader(f)
    batch = []

    for row in reader:
        predicate, *args = row
        fact = Fact(predicate=predicate.strip(),
                   args=[arg.strip() for arg in args])
        batch.append(fact)

        if len(batch) >= batch_size:
            for fact in batch:
                engine.insert_fact(fact)
            print(f"Inserted {len(batch)} facts...")
            batch = []

    # Insert remaining
    for fact in batch:
        engine.insert_fact(fact)

print(f"Total facts: {engine.kb.count()}")
```

### Multi-Query Analysis

```python
queries = [
    Query(predicate="parent", args=["alice", None]),
    Query(predicate="parent", args=["bob", None]),
    Query(predicate="parent", args=["carol", None]),
]

all_results = {}
for query in queries:
    bound_entity = [arg for arg in query.args if arg is not None][0]
    result = engine.query(query, k=10)
    all_results[bound_entity] = result.results

# Analyze
for entity, results in all_results.items():
    print(f"{entity} has {len(results)} children")
```

### Programmatic VSAR Generation

```python
# Generate VSAR program from data
def generate_vsar_program(facts_list, output_path):
    lines = []
    lines.append("@model FHRR(dim=8192, seed=42);")
    lines.append("@threshold(value=0.22);")
    lines.append("")

    for predicate, arg1, arg2 in facts_list:
        lines.append(f"fact {predicate}({arg1}, {arg2}).")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

# Use it
facts_data = [
    ("parent", "alice", "bob"),
    ("parent", "bob", "carol"),
]
generate_vsar_program(facts_data, "generated.vsar")
```

### Integration with DataFrames

```python
import pandas as pd

# Load from DataFrame
df = pd.DataFrame({
    "parent": ["alice", "bob", "carol"],
    "child": ["bob", "carol", "dave"]
})

for _, row in df.iterrows():
    fact = Fact(predicate="parent", args=[row["parent"], row["child"]])
    engine.insert_fact(fact)

# Query and return as DataFrame
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=10)

result_df = pd.DataFrame(result.results, columns=["entity", "score"])
print(result_df)
```

### Error Handling

```python
from pathlib import Path

try:
    # Load KB with validation
    kb_path = Path("family.h5")
    if not kb_path.exists():
        raise FileNotFoundError(f"KB file not found: {kb_path}")

    engine.load_kb(kb_path)

    # Query with validation
    query = Query(predicate="parent", args=["alice", None])

    # Check predicate exists
    if "parent" not in engine.kb.predicates():
        raise ValueError(f"Predicate 'parent' not found in KB")

    result = engine.query(query, k=10)

    if not result.results:
        print("Warning: Query returned no results")

except FileNotFoundError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Complete Application Example

```python
#!/usr/bin/env python3
"""
Company directory application using VSAR
"""

from pathlib import Path
from vsar.language.ast import Directive, Fact, Query
from vsar.language.loader import load_csv
from vsar.semantics.engine import VSAREngine

class CompanyDirectory:
    def __init__(self, kb_path: Path | None = None):
        """Initialize company directory."""
        directives = [
            Directive(name="model",
                     params={"type": "FHRR", "dim": 8192, "seed": 42}),
            Directive(name="threshold", params={"value": 0.22})
        ]
        self.engine = VSAREngine(directives)

        if kb_path and kb_path.exists():
            self.engine.load_kb(kb_path)

    def load_employees(self, csv_path: Path):
        """Load employee data from CSV."""
        facts = load_csv(csv_path)
        for fact in facts:
            self.engine.insert_fact(fact)

    def find_employees_in_department(self, department: str, k: int = 20):
        """Find all employees in a department."""
        query = Query(predicate="works_in", args=[None, department])
        result = self.engine.query(query, k=k)
        return [(e, s) for e, s in result.results if s > 0.7]

    def find_department(self, employee: str):
        """Find employee's department."""
        query = Query(predicate="works_in", args=[employee, None])
        result = self.engine.query(query, k=5)
        if result.results:
            return result.results[0][0]  # Top result
        return None

    def find_manager(self, employee: str):
        """Find employee's manager."""
        query = Query(predicate="manages", args=[None, employee])
        result = self.engine.query(query, k=5)
        if result.results:
            return result.results[0][0]
        return None

    def find_reports(self, manager: str, k: int = 20):
        """Find all direct reports of a manager."""
        query = Query(predicate="manages", args=[manager, None])
        result = self.engine.query(query, k=k)
        return [(e, s) for e, s in result.results if s > 0.7]

    def save(self, kb_path: Path):
        """Save KB to file."""
        self.engine.save_kb(kb_path)

    def stats(self):
        """Get statistics."""
        return self.engine.stats()

# Usage
if __name__ == "__main__":
    # Create directory
    directory = CompanyDirectory()

    # Load data
    directory.load_employees(Path("employees.csv"))

    # Queries
    eng_employees = directory.find_employees_in_department("engineering")
    print(f"Engineering employees: {[e for e, _ in eng_employees]}")

    dept = directory.find_department("alice")
    print(f"Alice's department: {dept}")

    manager = directory.find_manager("bob")
    print(f"Bob's manager: {manager}")

    reports = directory.find_reports("alice")
    print(f"Alice's reports: {[e for e, _ in reports]}")

    # Save
    directory.save(Path("company.h5"))

    # Stats
    print("Statistics:", directory.stats())
```

---

## Next Steps

- **[Basic Usage Guide](basic-usage.md)** - Learn VSAR fundamentals
- **[File Formats Guide](file-formats.md)** - Data loading details
- **[KB Management Guide](kb-management.md)** - Persistence and export
- **[API Reference](../api/semantics.md)** - Complete API documentation
