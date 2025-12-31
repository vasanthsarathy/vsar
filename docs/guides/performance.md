# Performance Tuning

Guide to optimizing VSAR performance for your workload.

## Key Parameters

### 1. Beam Width

Controls candidate bindings in joins:

```prolog
@beam(width=50);    // Default
@beam(width=100);   // More candidates, slower but more complete
@beam(width=20);    // Fewer candidates, faster
```

**Guidelines:**
- Small KB (<1K facts): 20-50
- Medium KB (1K-10K): 50-100
- Large KB (>10K): 100-200

### 2. Novelty Threshold

Controls duplicate detection:

```prolog
@novelty(threshold=0.95);   // Default (balanced)
@novelty(threshold=0.99);   // Stricter (more facts, slower)
@novelty(threshold=0.90);   // Looser (fewer facts, faster)
```

### 3. Dimensionality

Higher dimensions = better separation:

```prolog
@model FHRR(dim=1024, seed=42);   // Balanced
@model FHRR(dim=2048, seed=42);   // Higher quality, more memory
@model FHRR(dim=512, seed=42);    // Faster, less memory
```

### 4. Max Iterations

Prevent infinite loops:

```python
from vsar.semantics.chaining import apply_rules
result = apply_rules(engine, rules, max_iterations=100)
```

## Performance Metrics

| Facts | Query Time | Chaining (10 rules) |
|-------|------------|---------------------|
| 10^3  | <50ms      | <200ms              |
| 10^4  | <100ms     | <500ms              |
| 10^5  | <300ms     | <2s                 |
| 10^6  | <800ms     | <10s                |

*AMD EPYC 7742, dim=1024, beam=50*

## Optimization Tips

### 1. Use Semi-Naive Evaluation

Automatically enabled:

```python
apply_rules(engine, rules, semi_naive=True)  # Default
```

### 2. Monitor Progress

```python
result = apply_rules(engine, rules, max_iterations=100)
print(f"Iterations: {result.iterations}")
print(f"Per iteration: {result.derived_per_iteration}")
```

### 3. Profile Your Rules

```python
import time
start = time.time()
result = apply_rules(engine, rules)
print(f"Chaining took: {time.time() - start:.2f}s")
```

### 4. Reduce Dimensionality for Speed

```prolog
@model FHRR(dim=512, seed=42);  // 2x faster than 1024
```

## Memory Usage

- Base: ~50MB (dim=1024)
- Per 1000 facts: ~5MB
- Scales linearly

## Next Steps

- **[Rules & Chaining Guide](rules-and-chaining.md)** - Deep dive
- **[Examples](../examples.md)** - Optimized programs
- **[Capabilities](../capabilities.md)** - Performance characteristics
