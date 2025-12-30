# Language API Reference

The language layer provides parsing, AST structures, and loaders for VSAR programs.

## Module: `vsar.language.ast`

AST (Abstract Syntax Tree) node classes for VSAR programs.

### Directive

::: vsar.language.ast.Directive
    options:
      show_source: true
      heading_level: 4

Configuration directive for VSAR engine.

**Attributes:**

- `name` (str): Directive name ("model", "threshold", "beam")
- `params` (dict[str, Any]): Parameters as key-value pairs

**Example:**

```python
from vsar.language.ast import Directive

# Model directive
model_dir = Directive(
    name="model",
    params={"type": "FHRR", "dim": 8192, "seed": 42}
)

# Threshold directive
threshold_dir = Directive(
    name="threshold",
    params={"value": 0.22}
)
```

---

### Fact

::: vsar.language.ast.Fact
    options:
      show_source: true
      heading_level: 4

Ground fact (all arguments are constants).

**Attributes:**

- `predicate` (str): Predicate name
- `args` (list[str]): Argument list (all constants)

**Example:**

```python
from vsar.language.ast import Fact

# Binary fact
fact1 = Fact(predicate="parent", args=["alice", "bob"])

# Ternary fact
fact2 = Fact(predicate="transfer", args=["alice", "bob", "money"])

# Unary fact
fact3 = Fact(predicate="person", args=["alice"])
```

---

### Query

::: vsar.language.ast.Query
    options:
      show_source: true
      heading_level: 4

Query with exactly one variable (Phase 1).

**Attributes:**

- `predicate` (str): Predicate name
- `args` (list[str | None]): Arguments (None = variable)

**Methods:**

#### get_variables() -> list[int]

Returns positions of variables (0-indexed).

```python
query = Query(predicate="parent", args=["alice", None])
positions = query.get_variables()
# Returns: [1]
```

#### get_bound_args() -> dict[str, str]

Returns bound arguments as dict (1-indexed positions).

```python
query = Query(predicate="parent", args=["alice", None])
bound = query.get_bound_args()
# Returns: {"1": "alice"}
```

**Example:**

```python
from vsar.language.ast import Query

# Find children of alice
query1 = Query(predicate="parent", args=["alice", None])

# Find parents of bob
query2 = Query(predicate="parent", args=[None, "bob"])

# Ternary query
query3 = Query(predicate="transfer", args=["alice", None, "money"])
```

---

### Program

::: vsar.language.ast.Program
    options:
      show_source: true
      heading_level: 4

Complete VSAR program.

**Attributes:**

- `directives` (list[Directive]): Configuration directives
- `facts` (list[Fact]): Ground facts
- `queries` (list[Query]): Queries
- `rules` (list[Rule]): Rules (Phase 2, not yet used)

**Example:**

```python
from vsar.language.ast import Program, Directive, Fact, Query

program = Program(
    directives=[
        Directive(name="model",
                 params={"type": "FHRR", "dim": 8192, "seed": 42})
    ],
    facts=[
        Fact(predicate="parent", args=["alice", "bob"]),
        Fact(predicate="parent", args=["bob", "carol"])
    ],
    queries=[
        Query(predicate="parent", args=["alice", None])
    ],
    rules=[]
)
```

---

## Module: `vsar.language.parser`

VSARL parser using Lark.

### parse

::: vsar.language.parser.parse
    options:
      show_source: true
      heading_level: 4

Parse VSAR program from string.

**Parameters:**

- `text` (str): VSAR program text

**Returns:**

- `Program`: Parsed program

**Raises:**

- `ParseError`: If syntax is invalid

**Example:**

```python
from vsar.language.parser import parse

program_text = """
@model FHRR(dim=8192, seed=42);

fact parent(alice, bob).
query parent(alice, X)?
"""

program = parse(program_text)
print(f"Directives: {len(program.directives)}")
print(f"Facts: {len(program.facts)}")
print(f"Queries: {len(program.queries)}")
```

---

### parse_file

::: vsar.language.parser.parse_file
    options:
      show_source: true
      heading_level: 4

Parse VSAR program from file.

**Parameters:**

- `path` (Path | str): Path to .vsar file

**Returns:**

- `Program`: Parsed program

**Example:**

```python
from vsar.language.parser import parse_file
from pathlib import Path

program = parse_file(Path("family.vsar"))
```

---

## Module: `vsar.language.loader`

Loaders for CSV, JSONL, and VSAR files.

### load_csv

::: vsar.language.loader.load_csv
    options:
      show_source: true
      heading_level: 4

Load facts from CSV file.

**Parameters:**

- `path` (Path | str): Path to CSV file
- `predicate` (str | None): Predicate name (if CSV has no predicate column)

**Returns:**

- `list[Fact]`: List of facts

**Format 1: With predicate column**

```csv
parent,alice,bob
parent,bob,carol
lives_in,alice,boston
```

```python
facts = load_csv("facts.csv")
```

**Format 2: Without predicate column**

```csv
alice,bob
bob,carol
```

```python
facts = load_csv("parents.csv", predicate="parent")
```

---

### load_jsonl

::: vsar.language.loader.load_jsonl
    options:
      show_source: true
      heading_level: 4

Load facts from JSONL file.

**Parameters:**

- `path` (Path | str): Path to JSONL file

**Returns:**

- `list[Fact]`: List of facts

**Format:**

```jsonl
{"predicate": "parent", "args": ["alice", "bob"]}
{"predicate": "lives_in", "args": ["alice", "boston"]}
```

**Example:**

```python
facts = load_jsonl("facts.jsonl")
for fact in facts:
    print(f"{fact.predicate}({', '.join(fact.args)})")
```

---

### load_vsar

::: vsar.language.loader.load_vsar
    options:
      show_source: true
      heading_level: 4

Load complete VSAR program from file.

**Parameters:**

- `path` (Path | str): Path to .vsar file

**Returns:**

- `Program`: Complete program

**Example:**

```python
program = load_vsar("family.vsar")
print(f"Directives: {program.directives}")
print(f"Facts: {program.facts}")
print(f"Queries: {program.queries}")
```

---

### load_facts

::: vsar.language.loader.load_facts
    options:
      show_source: true
      heading_level: 4

Load facts with automatic format detection.

**Parameters:**

- `path` (Path | str): Path to file
- `format` (str): Format ("auto", "csv", "jsonl", "vsar")
- `predicate` (str | None): Predicate for CSV without predicate column

**Returns:**

- `list[Fact]`: List of facts

**Example:**

```python
# Auto-detect from extension
facts = load_facts("data.csv", format="auto")
facts = load_facts("data.jsonl", format="auto")

# Explicit format
facts = load_facts("data.txt", format="csv")
```

---

## Usage Examples

### Example 1: Parse VSAR File

```python
from vsar.language.loader import load_vsar
from vsar.semantics.engine import VSAREngine

# Load program
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

### Example 2: Load CSV and Query

```python
from vsar.language.loader import load_csv
from vsar.language.ast import Directive, Query
from vsar.semantics.engine import VSAREngine

# Load facts
facts = load_csv("people.csv")

# Create engine
directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert facts
for fact in facts:
    engine.insert_fact(fact)

# Query
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query, k=5)

for entity, score in result.results:
    print(f"{entity}: {score:.4f}")
```

### Example 3: Programmatic AST Construction

```python
from vsar.language.ast import Program, Directive, Fact, Query

# Build program programmatically
program = Program(
    directives=[
        Directive(name="model",
                 params={"type": "FHRR", "dim": 8192, "seed": 42}),
        Directive(name="threshold", params={"value": 0.22})
    ],
    facts=[
        Fact(predicate="parent", args=["alice", "bob"]),
        Fact(predicate="parent", args=["alice", "carol"]),
        Fact(predicate="parent", args=["bob", "dave"]),
    ],
    queries=[
        Query(predicate="parent", args=["alice", None]),
        Query(predicate="parent", args=[None, "dave"])
    ],
    rules=[]
)

print(f"Program has {len(program.facts)} facts and {len(program.queries)} queries")
```

---

## See Also

- [Semantics API](semantics.md) - VSAREngine
- [Trace API](trace.md) - Trace layer
- [Language Reference](../language-reference.md) - VSARL syntax
