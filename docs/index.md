# VSAR Documentation

VSAR (VSAX Reasoner) is a VSA-grounded reasoning tool that provides a unified inference substrate using hypervector algebra for deductive reasoning, approximate joins, and explainable results.

## Key Features

- **Deductive reasoning** over facts and Horn-style rules (Datalog-like)
- **Approximate unification** via VSA binding/unbinding and similarity search
- **Explainable results** with similarity scores, retrieved candidates, and rule firings
- **Dual-mode support**: Both classical VSA and Clifford algebra modes

## Quick Links

- [Getting Started](getting-started.md)
- [Architecture](architecture.md)
- [API Reference](api/index.md)

## Installation

```bash
pip install uv
uv init
uv add vsar
```

## Project Status

Phase 0 (Foundation) - Currently implementing the core kernel layer with VSA and Clifford backends.
