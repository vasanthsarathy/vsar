# Trace API Reference

The trace layer provides explanation DAGs for query execution.

## Module: `vsar.trace.events`

Trace event classes for explanation DAG.

### TraceEvent

::: vsar.trace.events.TraceEvent
    options:
      show_source: true
      heading_level: 4

Single trace event in the explanation DAG.

**Attributes:**

- `id` (str): Unique event ID (UUID)
- `type` (str): Event type ("query", "retrieval", etc.)
- `payload` (dict[str, Any]): Event-specific data
- `parent_ids` (list[str]): IDs of parent events
- `timestamp` (float): Unix timestamp

**Example:**

```python
from vsar.trace.events import TraceEvent

event = TraceEvent(
    id="abc123",
    type="query",
    payload={"predicate": "parent", "args": ["alice", None]},
    parent_ids=[],
    timestamp=1705334400.0
)
```

---

#### `to_dict() -> dict[str, Any]`

Convert event to dictionary.

**Returns:**

- dict with all event fields

**Example:**

```python
event_dict = event.to_dict()
# {
#   "id": "abc123",
#   "type": "query",
#   "payload": {...},
#   "parent_ids": [],
#   "timestamp": 1705334400.0
# }
```

---

#### `from_dict(data: dict[str, Any]) -> TraceEvent`

Create event from dictionary.

**Parameters:**

- `data` (dict): Event dictionary

**Returns:**

- TraceEvent instance

**Example:**

```python
data = {
    "id": "abc123",
    "type": "query",
    "payload": {"predicate": "parent"},
    "parent_ids": [],
    "timestamp": 1705334400.0
}
event = TraceEvent.from_dict(data)
```

---

## Module: `vsar.trace.collector`

Trace collector for building explanation DAGs.

### TraceCollector

::: vsar.trace.collector.TraceCollector
    options:
      show_source: true
      heading_level: 4

Collects trace events and builds DAG.

**Example:**

```python
from vsar.trace.collector import TraceCollector

collector = TraceCollector()

# Record events
query_id = collector.record("query", {"predicate": "parent"})
retrieval_id = collector.record("retrieval",
                                {"results": 5},
                                parent_ids=[query_id])

# Get DAG
dag = collector.get_dag()
```

---

#### `__init__()`

Initialize empty trace collector.

**Example:**

```python
from vsar.trace.collector import TraceCollector

collector = TraceCollector()
```

---

#### `record(event_type: str, payload: dict[str, Any], parent_ids: list[str] | None = None) -> str`

Record a trace event.

**Parameters:**

- `event_type` (str): Event type
- `payload` (dict): Event-specific data
- `parent_ids` (list[str] | None): Parent event IDs

**Returns:**

- str: New event ID

**Example:**

```python
# Root event (no parents)
query_id = collector.record("query", {
    "predicate": "parent",
    "args": ["alice", None]
})

# Child event
retrieval_id = collector.record("retrieval", {
    "num_results": 5,
    "results": [("bob", 0.92), ("carol", 0.91)]
}, parent_ids=[query_id])
```

---

#### `get_dag() -> list[TraceEvent]`

Get complete trace DAG.

**Returns:**

- list[TraceEvent]: All events in chronological order

**Example:**

```python
dag = collector.get_dag()

for event in dag:
    print(f"{event.type}: {event.payload}")
```

---

#### `get_subgraph(event_id: str) -> list[TraceEvent]`

Get event and all ancestors.

**Parameters:**

- `event_id` (str): Event ID

**Returns:**

- list[TraceEvent]: Event and all ancestors

**Example:**

```python
# Get subgraph for specific query
query_id = result.trace_id
subgraph = collector.get_subgraph(query_id)

print(f"Subgraph has {len(subgraph)} events")
for event in subgraph:
    print(f"  {event.type}")
```

---

#### `to_dict() -> dict[str, Any]`

Convert entire trace to dictionary.

**Returns:**

- dict with "events" key containing list of event dicts

**Example:**

```python
trace_dict = collector.to_dict()
# {
#   "events": [
#     {"id": "...", "type": "query", ...},
#     {"id": "...", "type": "retrieval", ...}
#   ]
# }

import json
with open("trace.json", "w") as f:
    json.dump(trace_dict, f, indent=2)
```

---

## Event Types

### Query Event

Recorded when a query is executed.

**Type:** `"query"`

**Payload:**

- `predicate` (str): Predicate name
- `args` (list): Query arguments
- `variables` (list[int]): Variable positions
- `bound_args` (dict): Bound arguments

**Example:**

```python
{
    "predicate": "parent",
    "args": ["alice", None],
    "variables": [1],
    "bound_args": {"1": "alice"}
}
```

---

### Retrieval Event

Recorded after retrieval completes.

**Type:** `"retrieval"`

**Payload:**

- `predicate` (str): Predicate queried
- `var_position` (int): Variable position (1-indexed)
- `k` (int): Number of results requested
- `num_results` (int): Actual results returned
- `results` (list): Top 5 results (truncated)

**Example:**

```python
{
    "predicate": "parent",
    "var_position": 2,
    "k": 10,
    "num_results": 2,
    "results": [("bob", 0.9234), ("carol", 0.9156)]
}
```

---

## Usage Examples

### Example 1: Basic Tracing

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert facts
engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))

# Query (automatically traced)
query = Query(predicate="parent", args=["alice", None])
result = engine.query(query)

# Access trace
trace = engine.trace.get_dag()
print(f"Total trace events: {len(trace)}")

for event in trace:
    print(f"{event.type}: {event.payload}")
```

### Example 2: Subgraph Analysis

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

# Execute multiple queries
queries = [
    Query(predicate="test", args=["a0", None]),
    Query(predicate="test", args=["a1", None]),
    Query(predicate="test", args=[None, "b0"]),
]

results = []
for query in queries:
    result = engine.query(query, k=5)
    results.append(result)

# Analyze each query's trace
for i, result in enumerate(results):
    subgraph = engine.trace.get_subgraph(result.trace_id)
    print(f"\nQuery {i} trace:")
    for event in subgraph:
        print(f"  {event.type}: {event.id[:8]}...")
```

### Example 3: Export Trace

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine
import json

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Insert and query
engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
result = engine.query(Query(predicate="parent", args=["alice", None]))

# Export trace
trace_dict = engine.trace.to_dict()

with open("trace.json", "w") as f:
    json.dump(trace_dict, f, indent=2)

print(f"Exported {len(trace_dict['events'])} trace events")
```

### Example 4: Trace Statistics

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

# Execute many queries
for i in range(100):
    engine.insert_fact(Fact(predicate="test", args=[f"a{i}", f"b{i}"]))

for i in range(10):
    query = Query(predicate="test", args=[f"a{i}", None])
    engine.query(query, k=5)

# Analyze trace
trace = engine.trace.get_dag()

# Count by type
from collections import Counter
event_types = Counter(e.type for e in trace)

print("Trace statistics:")
for event_type, count in event_types.items():
    print(f"  {event_type}: {count}")

# Analyze timestamps
timestamps = [e.timestamp for e in trace]
duration = max(timestamps) - min(timestamps)
print(f"Total execution time: {duration:.4f}s")
```

### Example 5: DAG Visualization (Text)

```python
from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine

directives = [
    Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
]
engine = VSAREngine(directives)

engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
result = engine.query(Query(predicate="parent", args=["alice", None]))

# Visualize DAG
trace = engine.trace.get_dag()

print("Trace DAG:")
for event in trace:
    indent = "  " * len(event.parent_ids)
    parent_info = f" (parents: {len(event.parent_ids)})" if event.parent_ids else ""
    print(f"{indent}{event.type}{parent_info}")
    print(f"{indent}  ID: {event.id[:8]}...")
    print(f"{indent}  Payload: {list(event.payload.keys())}")
```

---

## See Also

- [Semantics API](semantics.md) - VSAREngine
- [Language API](language.md) - AST and loaders
- [Getting Started](../getting-started.md) - Trace examples
