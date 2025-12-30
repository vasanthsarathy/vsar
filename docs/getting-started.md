# Getting Started with VSAR

This guide will help you get started with VSAR, from installation to running your first queries.

## Prerequisites

- Python 3.11 or higher
- pip (comes with Python)

## Installation

### Option 1: Install from PyPI (Recommended)

For most users, install VSAR directly from PyPI:

```bash
pip install vsar

# Verify installation
vsar --help
```

### Option 2: Development Install

For contributors or those working from source:

```bash
# Install uv
pip install uv

# Clone repository
git clone https://github.com/vasanthsarathy/vsar.git
cd vsar

# Install dependencies
uv sync

# For development, use uv run
uv run vsar --help
```

## Your First VSAR Program

Let's create a simple family tree knowledge base.

### Step 1: Create a VSAR Program

Create a file called `family.vsar`:

```prolog
@model FHRR(dim=8192, seed=42);
@threshold(value=0.22);

fact parent(alice, bob).
fact parent(alice, carol).
fact parent(bob, dave).
fact parent(carol, eve).

query parent(alice, X)?
query parent(X, dave)?
```

### Step 2: Run the Program

```bash
vsar run family.vsar
```

You should see output like:

```
Inserted 4 facts

┌─────────────────────────┐
│ Query: parent(alice, X) │
├────────┬────────────────┤
│ Entity │ Score          │
├────────┼────────────────┤
│ bob    │ 0.9234         │
│ carol  │ 0.9156         │
└────────┴────────────────┘

┌───────────────────────┐
│ Query: parent(X, dave) │
├────────┬──────────────┤
│ Entity │ Score        │
├────────┼──────────────┤
│ bob    │ 0.8876       │
└────────┴──────────────┘
```

### Understanding the Program

- `@model FHRR(dim=8192, seed=42);` - Configures the VSA backend with 8192 dimensions
- `@threshold(value=0.22);` - Sets the similarity threshold for retrieval
- `fact parent(alice, bob).` - Declares a ground fact
- `query parent(alice, X)?` - Queries for all children of alice (X is a variable)

## Working with CSV Files

VSAR can ingest facts from CSV files.

### Step 1: Create a CSV File

Create `people.csv`:

```csv
parent,alice,bob
parent,alice,carol
parent,bob,dave
parent,carol,eve
lives_in,alice,boston
lives_in,bob,cambridge
works_at,alice,mit
works_at,bob,harvard
```

### Step 2: Ingest the Facts

```bash
vsar ingest people.csv --kb family.h5
```

### Step 3: Query the Knowledge Base

Create `queries.vsar`:

```prolog
@model FHRR(dim=8192, seed=42);

query lives_in(X, boston)?
query works_at(alice, X)?
```

Run it:

```bash
vsar run queries.vsar
```

## Working with JSONL Files

VSAR also supports JSONL (JSON Lines) format.

Create `facts.jsonl`:

```jsonl
{"predicate": "parent", "args": ["alice", "bob"]}
{"predicate": "parent", "args": ["alice", "carol"]}
{"predicate": "lives_in", "args": ["alice", "boston"]}
```

Ingest it:

```bash
vsar ingest facts.jsonl --kb family.h5
```

## Knowledge Base Management

### Saving a Knowledge Base

After ingesting facts, save them to an HDF5 file:

```bash
vsar ingest people.csv --kb family.h5
```

### Inspecting a Knowledge Base

View statistics about a saved KB:

```bash
vsar inspect family.h5
```

Output:

```
Knowledge Base Statistics:
Total facts: 8
Predicates:
  - parent: 4 facts
  - lives_in: 2 facts
  - works_at: 2 facts
```

### Exporting a Knowledge Base

Export to JSON:

```bash
vsar export family.h5 --format json --output facts.json
```

Export to JSONL:

```bash
vsar export family.h5 --format jsonl --output facts.jsonl
```

## Advanced Features

### JSON Output for Scripting

Get machine-readable JSON output:

```bash
vsar run family.vsar --json
```

Output:

```json
{
  "queries": [
    {
      "predicate": "parent",
      "args": ["alice", null],
      "results": [
        {"entity": "bob", "score": 0.9234},
        {"entity": "carol", "score": 0.9156}
      ]
    }
  ]
}
```

### Showing Trace Information

View the explanation DAG for debugging:

```bash
vsar run family.vsar --trace
```

### Limiting Results

Limit the number of results per query:

```bash
vsar run family.vsar --k 10
```

## Using the Python API

For programmatic access, use the Python API:

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

# Create engine
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert facts
facts = [
    Fact(predicate="parent", args=["alice", "bob"]),
    Fact(predicate="parent", args=["alice", "carol"]),
]
for fact in facts:
    engine.insert_fact(fact)

# Execute query
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=5)

# Print results
for entity, score in result.results:
    print(f"{entity}: {score:.4f}")
```

## Next Steps

- **[CLI Reference](cli-reference.md)** - Learn all CLI commands and options
- **[Language Reference](language-reference.md)** - Complete VSARL syntax guide
- **[User Guides](guides/basic-usage.md)** - Step-by-step tutorials
- **[Python API Guide](guides/python-api.md)** - Build applications with VSAR
- **[Architecture](architecture.md)** - Understand how VSAR works

## Common Workflows

### Workflow 1: Quick Exploration

```bash
# Create a .vsar program with facts and queries
nano explore.vsar

# Run it
vsar run explore.vsar
```

### Workflow 2: Large Dataset Ingestion

```bash
# Ingest from CSV
vsar ingest large_dataset.csv --kb data.h5

# Query it
vsar run queries.vsar

# Export results
vsar export data.h5 --format json --output results.json
```

### Workflow 3: Programmatic Integration

```python
from vsar.language.loader import load_csv
from vsar.semantics.engine import VSAREngine
from vsar.language.ast import Directive, Query

# Load facts from CSV
facts = load_csv("data.csv")

# Create engine
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert all facts
for fact in facts:
    engine.insert_fact(fact)

# Query programmatically
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=10)

# Process results
for entity, score in result.results:
    if score > 0.8:
        print(f"High-confidence result: {entity} ({score:.4f})")
```

## Getting Help

If you run into issues:

1. Check the [CLI Reference](cli-reference.md) for command syntax
2. Review the [Language Reference](language-reference.md) for VSARL syntax
3. Read the [User Guides](guides/basic-usage.md) for detailed examples
4. [Open an issue](https://github.com/vasanthsarathy/vsar/issues) on GitHub
