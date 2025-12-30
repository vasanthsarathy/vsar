# Knowledge Base Management Guide

This guide covers saving, loading, exporting, and inspecting knowledge bases in VSAR.

## Table of Contents

1. [HDF5 Persistence](#hdf5-persistence)
2. [Saving Knowledge Bases](#saving-knowledge-bases)
3. [Loading Knowledge Bases](#loading-knowledge-bases)
4. [Exporting Knowledge Bases](#exporting-knowledge-bases)
5. [Inspecting Knowledge Bases](#inspecting-knowledge-bases)
6. [Best Practices](#best-practices)

---

## HDF5 Persistence

VSAR uses HDF5 format for persistent storage of knowledge bases. HDF5 provides:

- **Efficient storage**: Compressed binary format
- **Fast access**: Direct access to predicate bundles
- **Portability**: Cross-platform compatibility
- **Metadata**: Stores VSA configuration (dim, seed, backend)

### File Extension

VSAR knowledge bases use the `.h5` extension:

```bash
family.h5        # Knowledge base file
company.h5       # Another KB
large_kb.h5      # Large-scale KB
```

---

## Saving Knowledge Bases

### Method 1: CLI with Ingest

Automatically save during ingestion:

```bash
vsar ingest facts.csv --kb family.h5
```

This creates `family.h5` with all facts from `facts.csv`.

### Method 2: Python API

```python
from vsar.language.ast import Directive, Fact
from vsar.semantics.engine import VSAREngine

# Create engine
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert facts
facts = [
    Fact(predicate="parent", args=["alice", "bob"]),
    Fact(predicate="parent", args=["bob", "carol"]),
]
for fact in facts:
    engine.insert_fact(fact)

# Save KB
engine.save_kb("family.h5")
print("Knowledge base saved to family.h5")
```

### What Gets Saved?

When you save a KB, VSAR stores:

1. **All facts** - Predicate bundles and metadata
2. **VSA configuration** - Backend type, dimensions, seed
3. **Symbol registry** - Basis vectors for entities and predicates
4. **Predicate index** - Fast lookup by predicate name

**Not saved:**

- Queries (these are ephemeral)
- Trace information (generated during execution)
- Directives (configuration embedded in KB metadata)

---

## Loading Knowledge Bases

### Method 1: CLI

Load a KB and run queries:

```bash
# Create query file
cat > queries.vsar << 'EOF'
@model FHRR(dim=8192, seed=42);
query parent(alice, X)?
EOF

# Note: KB is loaded automatically by the engine when needed
vsar run queries.vsar
```

### Method 2: Python API

```python
from vsar.language.ast import Directive, Query
from vsar.semantics.engine import VSAREngine

# Create engine with SAME configuration as when KB was saved
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Load KB
engine.load_kb("family.h5")

# Query the loaded KB
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=5)

for entity, score in result.results:
    print(f"{entity}: {score:.4f}")
```

### Important: Configuration Matching

**The engine configuration MUST match the KB's configuration:**

```python
# KB was created with dim=8192, seed=42

# ✓ Correct: Same configuration
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]

# ✗ Wrong: Different dimensions
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 4096, "seed": 42})
]

# ✗ Wrong: Different backend
directives = [
    Directive(name="model", params={"type": "MAP", "dim": 8192, "seed": 42})
]
```

If configurations don't match, results will be incorrect or queries may fail.

---

## Exporting Knowledge Bases

Export KBs to human-readable formats for analysis or sharing.

### Export to JSON

**CLI:**

```bash
vsar export family.h5 --format json --output facts.json
```

**Output (facts.json):**

```json
{
  "parent": [
    {"args": ["alice", "bob"]},
    {"args": ["alice", "carol"]},
    {"args": ["bob", "dave"]}
  ],
  "lives_in": [
    {"args": ["alice", "boston"]},
    {"args": ["bob", "cambridge"]}
  ]
}
```

**Python:**

```python
from vsar.semantics.engine import VSAREngine
from vsar.language.ast import Directive
import json

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)
engine.load_kb("family.h5")

data = engine.export_kb("json")

with open("facts.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Export to JSONL

**CLI:**

```bash
vsar export family.h5 --format jsonl --output facts.jsonl
```

**Output (facts.jsonl):**

```jsonl
{"predicate": "parent", "args": ["alice", "bob"]}
{"predicate": "parent", "args": ["alice", "carol"]}
{"predicate": "parent", "args": ["bob", "dave"]}
{"predicate": "lives_in", "args": ["alice", "boston"]}
{"predicate": "lives_in", "args": ["bob", "cambridge"]}
```

**Python:**

```python
data = engine.export_kb("jsonl")

with open("facts.jsonl", "w") as f:
    f.write(data)
```

### Export to CSV

Convert JSONL to CSV:

```python
import json
import csv

# Export to JSONL first
data = engine.export_kb("jsonl")

# Convert to CSV
with open("facts.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for line in data.strip().split("\n"):
        obj = json.loads(line)
        writer.writerow([obj["predicate"]] + obj["args"])
```

---

## Inspecting Knowledge Bases

### CLI Inspection

Get statistics about a KB:

```bash
vsar inspect family.h5
```

**Output:**

```
Knowledge Base Statistics:
========================

Total facts: 8

Predicates:
  - parent: 4 facts
  - lives_in: 2 facts
  - works_at: 2 facts

VSA Configuration:
  - Backend: FHRR
  - Dimensions: 8192
  - Seed: 42
```

### Python Inspection

```python
from vsar.semantics.engine import VSAREngine
from vsar.language.ast import Directive

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)
engine.load_kb("family.h5")

# Get statistics
stats = engine.stats()

print(f"Total facts: {stats['total_facts']}")
print("Predicates:")
for predicate, count in stats['predicates'].items():
    print(f"  - {predicate}: {count} facts")
```

### Detailed Inspection

Get facts for specific predicate:

```python
# Get all parent facts
parent_facts = engine.kb.get_facts("parent")
print("Parent facts:")
for fact in parent_facts:
    print(f"  parent({', '.join(fact)})")

# List all predicates
predicates = engine.kb.predicates()
print(f"Available predicates: {predicates}")

# Count facts per predicate
for pred in predicates:
    count = engine.kb.count(pred)
    print(f"{pred}: {count} facts")
```

---

## Best Practices

### 1. Version Your KBs

Use descriptive names with versions:

```bash
company_v1.h5
company_v2.h5
company_2025_01_15.h5
```

### 2. Backup Important KBs

```bash
cp family.h5 family_backup_$(date +%Y%m%d).h5
```

### 3. Document Configuration

Record the configuration used to create a KB:

```bash
cat > family_kb_config.txt << 'EOF'
KB: family.h5
Created: 2025-01-15
Backend: FHRR
Dimensions: 8192
Seed: 42
Facts: 1000
EOF
```

### 4. Validate After Loading

```python
engine.load_kb("family.h5")

# Validate
stats = engine.stats()
expected_facts = 1000

if stats["total_facts"] != expected_facts:
    print(f"Warning: Expected {expected_facts} facts, got {stats['total_facts']}")
```

### 5. Export for Archives

Export to JSON for long-term archival:

```bash
vsar export family.h5 --format json --output family_archive.json
gzip family_archive.json
```

JSON is human-readable and portable, while H5 is optimized for VSAR.

---

## Complete Workflow Examples

### Workflow 1: Build, Save, Query

```bash
# Step 1: Ingest facts and save KB
vsar ingest company.csv --kb company.h5

# Step 2: Inspect KB
vsar inspect company.h5

# Step 3: Create queries
cat > queries.vsar << 'EOF'
@model FHRR(dim=8192, seed=42);
query works_in(X, engineering)?
query manages(alice, X)?
EOF

# Step 4: Run queries
vsar run queries.vsar

# Step 5: Export for analysis
vsar export company.h5 --format json --output company_export.json
```

### Workflow 2: Incremental Updates

```python
from vsar.language.ast import Directive, Fact
from vsar.language.loader import load_csv
from vsar.semantics.engine import VSAREngine

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]

# Day 1: Create KB
engine = VSAREngine(directives)
facts = load_csv("day1.csv")
for fact in facts:
    engine.insert_fact(fact)
engine.save_kb("kb_day1.h5")

# Day 2: Load and add more facts
engine2 = VSAREngine(directives)
engine2.load_kb("kb_day1.h5")
new_facts = load_csv("day2.csv")
for fact in new_facts:
    engine2.insert_fact(fact)
engine2.save_kb("kb_day2.h5")

# Day 3: Continue...
```

### Workflow 3: Migration

Migrate KB to different configuration:

```python
from vsar.language.ast import Directive
from vsar.semantics.engine import VSAREngine

# Load from old KB (dim=4096)
old_directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 4096, "seed": 42})
]
old_engine = VSAREngine(old_directives)
old_engine.load_kb("old_kb.h5")

# Export facts
data = old_engine.export_kb("jsonl")

# Create new KB with higher dimensions
new_directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
new_engine = VSAREngine(new_directives)

# Re-insert all facts
import json
for line in data.strip().split("\n"):
    obj = json.loads(line)
    from vsar.language.ast import Fact
    fact = Fact(predicate=obj["predicate"], args=obj["args"])
    new_engine.insert_fact(fact)

# Save new KB
new_engine.save_kb("new_kb.h5")
```

### Workflow 4: Merge Multiple KBs

```python
from vsar.language.ast import Directive
from vsar.semantics.engine import VSAREngine

# Configuration (must be same for all KBs)
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]

# Load KB 1
engine1 = VSAREngine(directives)
engine1.load_kb("kb1.h5")
facts1 = engine1.export_kb("jsonl")

# Load KB 2
engine2 = VSAREngine(directives)
engine2.load_kb("kb2.h5")
facts2 = engine2.export_kb("jsonl")

# Merge
merged_engine = VSAREngine(directives)

import json
from vsar.language.ast import Fact

for line in (facts1 + "\n" + facts2).strip().split("\n"):
    obj = json.loads(line)
    fact = Fact(predicate=obj["predicate"], args=obj["args"])
    merged_engine.insert_fact(fact)

# Save merged KB
merged_engine.save_kb("merged.h5")
print(f"Merged KB has {merged_engine.kb.count()} facts")
```

---

## Troubleshooting

### Problem: "Cannot load KB"

**Error:**

```
Error: Failed to load knowledge base from 'family.h5'
```

**Solutions:**

1. **Check file exists:**

```bash
ls -lh family.h5
```

2. **Check file integrity:**

```bash
h5dump -H family.h5  # Requires h5py tools
```

3. **Verify configuration matches:**

```python
# Configuration must match what was used to create KB
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
```

### Problem: "Wrong number of facts after loading"

**Cause:** KB was created with different configuration

**Solution:** Use exact same configuration:

```python
# When creating KB
directives_create = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]

# When loading KB (must be identical)
directives_load = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
```

### Problem: "Export file too large"

**Solution:** Use JSONL with compression:

```bash
vsar export large.h5 --format jsonl --output facts.jsonl
gzip facts.jsonl
```

---

## Next Steps

- **[Basic Usage Guide](basic-usage.md)** - Facts and queries
- **[File Formats Guide](file-formats.md)** - CSV, JSONL, VSAR formats
- **[Python API Guide](python-api.md)** - Programmatic usage
- **[CLI Reference](../cli-reference.md)** - Command-line options
