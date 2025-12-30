# CLI Reference

VSAR provides a command-line interface for executing programs, ingesting facts, and managing knowledge bases.

## Installation

After installing VSAR via pip, the `vsar` command is available globally:

```bash
pip install vsar
vsar --help
```

For development:

```bash
uv sync
uv run vsar --help
```

## Global Options

All commands support these global options:

- `--help` - Show help message and exit
- `--version` - Show VSAR version

## Commands

### `vsar run`

Execute a VSAR program file.

**Usage:**

```bash
vsar run PROGRAM_PATH [OPTIONS]
```

**Arguments:**

- `PROGRAM_PATH` - Path to the `.vsar` program file (required)

**Options:**

- `--json` - Output results as JSON instead of tables
- `--trace` - Show trace DAG for debugging
- `--k INTEGER` - Maximum number of results per query (default: 10)

**Examples:**

```bash
# Basic execution
vsar run family.vsar

# With JSON output for scripting
vsar run family.vsar --json

# Show trace information
vsar run family.vsar --trace

# Limit to 5 results per query
vsar run family.vsar --k 5

# Combine options
vsar run family.vsar --json --trace --k 20
```

**Output Format (Default):**

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
```

**Output Format (--json):**

```json
{
  "num_facts": 4,
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

---

### `vsar ingest`

Ingest facts from CSV or JSONL files into a knowledge base.

**Usage:**

```bash
vsar ingest FACTS_PATH [OPTIONS]
```

**Arguments:**

- `FACTS_PATH` - Path to facts file (.csv or .jsonl) (required)

**Options:**

- `--kb PATH` - Path to save/load knowledge base (HDF5 file)
- `--predicate TEXT` - Predicate name (for CSV without predicate column)
- `--format [auto|csv|jsonl]` - Input format (default: auto-detect)
- `--dim INTEGER` - VSA dimensionality (default: 8192)
- `--seed INTEGER` - Random seed for determinism (default: 42)

**Examples:**

**CSV with predicate column:**

```bash
# Format: predicate,arg1,arg2,...
vsar ingest facts.csv --kb family.h5
```

**CSV without predicate column:**

```bash
# Format: arg1,arg2,...
vsar ingest parents.csv --predicate parent --kb family.h5
```

**JSONL format:**

```bash
# Format: {"predicate": "parent", "args": ["alice", "bob"]}
vsar ingest facts.jsonl --kb family.h5
```

**Custom VSA parameters:**

```bash
vsar ingest large.csv --kb large.h5 --dim 16384 --seed 100
```

**Auto-format detection:**

```bash
# Extension determines format (.csv or .jsonl)
vsar ingest data.csv
vsar ingest data.jsonl
```

**Output:**

```
Ingesting facts from facts.csv...
Inserted 1000 facts into KB
Knowledge base saved to family.h5
```

---

### `vsar export`

Export knowledge base to JSON or JSONL format.

**Usage:**

```bash
vsar export KB_PATH [OPTIONS]
```

**Arguments:**

- `KB_PATH` - Path to knowledge base file (.h5) (required)

**Options:**

- `--output PATH` - Output file path (default: stdout)
- `--format [json|jsonl]` - Export format (default: json)

**Examples:**

**Export to JSON:**

```bash
vsar export family.h5 --format json --output facts.json
```

**Export to JSONL:**

```bash
vsar export family.h5 --format jsonl --output facts.jsonl
```

**Output to stdout:**

```bash
vsar export family.h5 --format json
```

**JSON Output Format:**

```json
{
  "parent": [
    {"args": ["alice", "bob"]},
    {"args": ["alice", "carol"]}
  ],
  "lives_in": [
    {"args": ["alice", "boston"]}
  ]
}
```

**JSONL Output Format:**

```jsonl
{"predicate": "parent", "args": ["alice", "bob"]}
{"predicate": "parent", "args": ["alice", "carol"]}
{"predicate": "lives_in", "args": ["alice", "boston"]}
```

---

### `vsar inspect`

Display statistics about a knowledge base.

**Usage:**

```bash
vsar inspect [KB_PATH]
```

**Arguments:**

- `KB_PATH` - Path to knowledge base file (.h5) (optional)

**Examples:**

**Inspect a saved KB:**

```bash
vsar inspect family.h5
```

**Inspect current session:**

```bash
vsar inspect
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

---

## Common Workflows

### Workflow 1: Create and Query KB

```bash
# Step 1: Create a VSAR program
cat > family.vsar << 'EOF'
@model FHRR(dim=8192, seed=42);

fact parent(alice, bob).
fact parent(bob, carol).

query parent(alice, X)?
EOF

# Step 2: Run it
vsar run family.vsar
```

### Workflow 2: Ingest CSV and Query

```bash
# Step 1: Create CSV
cat > people.csv << 'EOF'
parent,alice,bob
parent,bob,carol
lives_in,alice,boston
EOF

# Step 2: Ingest
vsar ingest people.csv --kb family.h5

# Step 3: Create query program
cat > queries.vsar << 'EOF'
@model FHRR(dim=8192, seed=42);
query parent(alice, X)?
query lives_in(X, boston)?
EOF

# Step 4: Run queries
vsar run queries.vsar

# Step 5: Inspect KB
vsar inspect family.h5

# Step 6: Export
vsar export family.h5 --format json --output facts.json
```

### Workflow 3: Large-Scale Processing

```bash
# Ingest 1M facts
vsar ingest large_dataset.csv --kb large.h5 --dim 16384

# Query with limited results
vsar run queries.vsar --k 10 --json > results.json

# Export for analysis
vsar export large.h5 --format jsonl --output export.jsonl
```

### Workflow 4: Scripting with JSON

```bash
# Get JSON output
vsar run family.vsar --json | jq '.queries[0].results[] | select(.score > 0.9)'

# Process multiple queries
for query_file in queries/*.vsar; do
    vsar run "$query_file" --json --k 5
done | jq -s '.'
```

---

## Error Handling

### Common Errors

**File not found:**

```bash
$ vsar run missing.vsar
Error: File 'missing.vsar' not found
```

**Invalid VSAR syntax:**

```bash
$ vsar run bad.vsar
Error: Parse error at line 3: Unexpected token 'X'
```

**KB file not found:**

```bash
$ vsar inspect missing.h5
Error: Knowledge base file 'missing.h5' not found
```

**Invalid format:**

```bash
$ vsar ingest data.txt
Error: Cannot auto-detect format for '.txt'. Use --format option.
```

---

## Exit Codes

- `0` - Success
- `1` - General error (file not found, parse error, etc.)
- `2` - Invalid arguments

---

## Environment Variables

VSAR respects these environment variables:

- `VSAR_KB_PATH` - Default path for knowledge base files
- `VSAR_DIM` - Default VSA dimensionality (default: 8192)
- `VSAR_SEED` - Default random seed (default: 42)

**Example:**

```bash
export VSAR_DIM=16384
export VSAR_SEED=100
vsar ingest facts.csv  # Uses VSAR_DIM and VSAR_SEED
```

---

## Performance Tips

1. **Use appropriate dimensions**: Larger dimensions (16384) provide better accuracy but slower queries
2. **Batch ingestion**: Ingest all facts at once rather than incrementally
3. **Limit results**: Use `--k` to limit results for faster queries
4. **JSON for scripting**: Use `--json` for programmatic processing
5. **Persistent KBs**: Save to HDF5 for repeated queries

**Benchmark (dim=8192):**

- 10^3 facts: <100ms per query
- 10^4 facts: <200ms per query
- 10^5 facts: <500ms per query
- 10^6 facts: <1s per query

---

## See Also

- [Getting Started](getting-started.md) - Quick start guide
- [Language Reference](language-reference.md) - VSARL syntax
- [User Guides](guides/basic-usage.md) - Detailed tutorials
- [Python API](guides/python-api.md) - Programmatic usage
