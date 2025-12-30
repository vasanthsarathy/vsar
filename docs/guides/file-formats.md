# File Formats Guide

VSAR supports multiple file formats for ingesting facts: CSV, JSONL, and native VSAR files. This guide covers all formats in detail.

## Table of Contents

1. [VSAR Format (.vsar)](#vsar-format)
2. [CSV Format (.csv)](#csv-format)
3. [JSONL Format (.jsonl)](#jsonl-format)
4. [Format Comparison](#format-comparison)
5. [Auto-Detection](#auto-detection)
6. [Best Practices](#best-practices)

---

## VSAR Format

The native VSAR format (`.vsar` files) is the most expressive format, supporting directives, facts, queries, and comments.

### Basic Structure

```prolog
// Directives (configuration)
@model FHRR(dim=8192, seed=42);
@threshold(value=0.22);

// Facts (knowledge)
fact parent(alice, bob).
fact parent(bob, carol).

// Queries (questions)
query parent(alice, X)?
```

### Complete Example

```prolog
/*
  Family Tree Knowledge Base
  Version: 1.0
*/

// Configure VSA backend
@model FHRR(dim=8192, seed=42);
@threshold(value=0.22);

// Parent relationships
fact parent(alice, bob).
fact parent(alice, carol).
fact parent(bob, dave).

// Location data
fact lives_in(alice, boston).
fact lives_in(bob, cambridge).

// Queries
query parent(alice, X)?
query lives_in(X, boston)?
```

### Loading VSAR Files

**CLI:**

```bash
vsar run family.vsar
```

**Python:**

```python
from vsar.language.loader import load_vsar

program = load_vsar("family.vsar")
print(f"Directives: {len(program.directives)}")
print(f"Facts: {len(program.facts)}")
print(f"Queries: {len(program.queries)}")
```

---

## CSV Format

CSV (Comma-Separated Values) is the simplest format for bulk fact ingestion.

### Format 1: With Predicate Column

The first column contains the predicate name, remaining columns are arguments.

**File: facts.csv**

```csv
parent,alice,bob
parent,alice,carol
parent,bob,dave
lives_in,alice,boston
lives_in,bob,cambridge
works_at,alice,mit
```

**Loading:**

```bash
vsar ingest facts.csv --kb family.h5
```

**Python:**

```python
from vsar.language.loader import load_csv

facts = load_csv("facts.csv")
for fact in facts:
    print(f"{fact.predicate}({', '.join(fact.args)})")
```

### Format 2: Without Predicate Column

All rows use the same predicate, specified via `--predicate` flag.

**File: parents.csv**

```csv
alice,bob
alice,carol
bob,dave
bob,eve
```

**Loading:**

```bash
vsar ingest parents.csv --predicate parent --kb family.h5
```

**Python:**

```python
from vsar.language.loader import load_csv

facts = load_csv("parents.csv", predicate="parent")
for fact in facts:
    print(f"{fact.predicate}({', '.join(fact.args)})")
```

### CSV Details

**Whitespace handling:**

```csv
parent,  alice  ,  bob
lives_in,alice,boston
```

Leading/trailing whitespace is automatically stripped.

**Headers:**

CSV files should NOT have headers. Every line is treated as a fact.

**Empty lines:**

Empty lines are skipped.

**Example with different arities:**

```csv
person,alice
person,bob
parent,alice,bob
transfer,alice,bob,money
```

---

## JSONL Format

JSONL (JSON Lines) is a structured format with one JSON object per line.

### Basic Structure

Each line is a JSON object with `predicate` and `args` fields:

```jsonl
{"predicate": "parent", "args": ["alice", "bob"]}
{"predicate": "parent", "args": ["bob", "carol"]}
{"predicate": "lives_in", "args": ["alice", "boston"]}
```

### Complete Example

**File: facts.jsonl**

```jsonl
{"predicate": "parent", "args": ["alice", "bob"]}
{"predicate": "parent", "args": ["alice", "carol"]}
{"predicate": "parent", "args": ["bob", "dave"]}
{"predicate": "sibling", "args": ["bob", "carol"]}
{"predicate": "lives_in", "args": ["alice", "boston"]}
{"predicate": "lives_in", "args": ["bob", "cambridge"]}
{"predicate": "works_at", "args": ["alice", "mit"]}
{"predicate": "transfer", "args": ["alice", "bob", "money"]}
```

### Loading JSONL Files

**CLI:**

```bash
vsar ingest facts.jsonl --kb family.h5
```

**Python:**

```python
from vsar.language.loader import load_jsonl

facts = load_jsonl("facts.jsonl")
for fact in facts:
    print(f"{fact.predicate}({', '.join(fact.args)})")
```

### JSONL with Metadata

JSONL supports additional metadata (ignored by VSAR):

```jsonl
{"predicate": "parent", "args": ["alice", "bob"], "confidence": 1.0, "source": "birth_certificate"}
{"predicate": "parent", "args": ["bob", "carol"], "confidence": 0.9, "source": "inference"}
```

VSAR only reads `predicate` and `args`; other fields are ignored.

---

## Format Comparison

| Feature | VSAR | CSV | JSONL |
|---------|------|-----|-------|
| **Directives** | ✓ | ✗ | ✗ |
| **Facts** | ✓ | ✓ | ✓ |
| **Queries** | ✓ | ✗ | ✗ |
| **Comments** | ✓ | ✗ | ✗ |
| **Metadata** | ✗ | ✗ | ✓ (ignored) |
| **Human-readable** | ✓✓ | ✓ | ✓ |
| **Bulk ingestion** | ✓ | ✓✓ | ✓✓ |
| **Mixed predicates** | ✓ | ✓ | ✓ |
| **Same predicate** | ✓ | ✓✓ | ✓ |

**Recommendations:**

- **VSAR**: Complete programs with queries
- **CSV**: Bulk ingestion, simple datasets
- **JSONL**: Structured data, external tools integration

---

## Auto-Detection

VSAR automatically detects format based on file extension:

```bash
vsar ingest facts.csv      # Auto-detects CSV
vsar ingest facts.jsonl    # Auto-detects JSONL
vsar run program.vsar      # Auto-detects VSAR
```

**Python:**

```python
from vsar.language.loader import load_facts

# Auto-detects format
facts = load_facts("facts.csv", format="auto")
facts = load_facts("facts.jsonl", format="auto")
```

**Explicit format:**

```python
facts = load_facts("data.txt", format="csv")
facts = load_facts("data.txt", format="jsonl")
```

---

## Best Practices

### When to Use VSAR Format

**Use VSAR when:**

- Writing complete programs with queries
- Need directives (@model, @threshold)
- Want inline comments for documentation
- Prototyping and exploration

**Example:**

```prolog
// Exploration: Testing different configurations
@model FHRR(dim=4096, seed=42);
@threshold(value=0.20);

fact parent(alice, bob).
fact parent(bob, carol).

query parent(alice, X)?  // Should return bob
query parent(X, carol)?  // Should return bob
```

### When to Use CSV Format

**Use CSV when:**

- Ingesting large datasets
- Data from spreadsheets or databases
- Simple, uniform facts
- All facts share same predicate

**Example:**

```bash
# Export from database
psql -c "COPY (SELECT parent_id, child_id FROM family) TO STDOUT CSV" > parents.csv

# Ingest into VSAR
vsar ingest parents.csv --predicate parent --kb family.h5
```

### When to Use JSONL Format

**Use JSONL when:**

- Integrating with JSON-based systems
- Need structured metadata
- Streaming data ingestion
- Mixed predicates with validation

**Example:**

```python
import json

# Generate from code
with open("facts.jsonl", "w") as f:
    for parent, child in relationships:
        fact = {"predicate": "parent", "args": [parent, child]}
        f.write(json.dumps(fact) + "\n")

# Ingest
vsar ingest facts.jsonl --kb family.h5
```

---

## Conversion Between Formats

### VSAR to CSV

Extract facts from VSAR file:

```python
from vsar.language.loader import load_vsar
import csv

program = load_vsar("family.vsar")

with open("facts.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for fact in program.facts:
        writer.writerow([fact.predicate] + fact.args)
```

### VSAR to JSONL

```python
from vsar.language.loader import load_vsar
import json

program = load_vsar("family.vsar")

with open("facts.jsonl", "w") as f:
    for fact in program.facts:
        obj = {"predicate": fact.predicate, "args": fact.args}
        f.write(json.dumps(obj) + "\n")
```

### CSV to JSONL

```python
import csv
import json

with open("facts.csv") as csvfile, open("facts.jsonl", "w") as jsonlfile:
    reader = csv.reader(csvfile)
    for row in reader:
        predicate, *args = row
        obj = {"predicate": predicate.strip(), "args": [arg.strip() for arg in args]}
        jsonlfile.write(json.dumps(obj) + "\n")
```

### JSONL to CSV

```python
import json
import csv

with open("facts.jsonl") as jsonlfile, open("facts.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for line in jsonlfile:
        obj = json.loads(line)
        writer.writerow([obj["predicate"]] + obj["args"])
```

---

## Advanced Examples

### Example 1: Mixed Arities in CSV

```csv
person,alice
person,bob
parent,alice,bob
parent,bob,carol
transfer,alice,bob,money
meeting,alice,bob,monday,2pm
```

All valid! VSAR handles different arities.

### Example 2: Large-Scale CSV Ingestion

```bash
# Generate 1M facts
python3 << 'EOF'
import csv

with open("large.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for i in range(1000000):
        writer.writerow(["relation", f"entity_{i}", f"value_{i}"])
EOF

# Ingest (takes ~10 seconds)
vsar ingest large.csv --kb large.h5 --dim 8192

# Query
vsar run queries.vsar
```

### Example 3: JSONL from API

```python
import requests
import json

# Fetch from API
response = requests.get("https://api.example.com/facts")
data = response.json()

# Convert to JSONL
with open("api_facts.jsonl", "w") as f:
    for item in data["results"]:
        fact = {
            "predicate": item["relation"],
            "args": [item["subject"], item["object"]]
        }
        f.write(json.dumps(fact) + "\n")

# Ingest
vsar ingest api_facts.jsonl --kb api.h5
```

### Example 4: Streaming Ingestion

```python
from vsar.language.ast import Fact, Directive
from vsar.semantics.engine import VSAREngine

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Stream from large JSONL file
import json

with open("large_facts.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        fact = Fact(predicate=obj["predicate"], args=obj["args"])
        engine.insert_fact(fact)

        # Progress tracking
        if engine.kb.count() % 10000 == 0:
            print(f"Inserted {engine.kb.count()} facts...")

print(f"Total facts: {engine.kb.count()}")
engine.save_kb("streamed.h5")
```

---

## Validation

### CSV Validation

```python
import csv

def validate_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 1):
            if not row:
                print(f"Line {i}: Empty row")
                continue
            if len(row) < 2:
                print(f"Line {i}: Too few columns (need predicate + args)")
            for j, field in enumerate(row):
                if not field.strip():
                    print(f"Line {i}, Column {j}: Empty field")

validate_csv("facts.csv")
```

### JSONL Validation

```python
import json

def validate_jsonl(path):
    with open(path) as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {i}: Invalid JSON - {e}")
                continue

            if "predicate" not in obj:
                print(f"Line {i}: Missing 'predicate' field")
            if "args" not in obj:
                print(f"Line {i}: Missing 'args' field")
            elif not isinstance(obj["args"], list):
                print(f"Line {i}: 'args' must be a list")

validate_jsonl("facts.jsonl")
```

---

## Next Steps

- **[Basic Usage Guide](basic-usage.md)** - Learn facts and queries
- **[KB Management Guide](kb-management.md)** - Save, load, export
- **[Python API Guide](python-api.md)** - Programmatic ingestion
- **[CLI Reference](../cli-reference.md)** - Command-line options
