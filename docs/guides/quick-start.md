# Quick Start

Get up and running with VSAR in 5 minutes.

## Install

```bash
pip install vsar
vsar --help
```

## Hello World

Create `hello.vsar`:

```prolog
@model FHRR(dim=1024, seed=42);

fact parent(alice, bob).
fact parent(bob, carol).

query parent(alice, X)?
```

Run:

```bash
vsar run hello.vsar
```

## With Rules

Create `reasoning.vsar`:

```prolog
@model FHRR(dim=1024, seed=42);
@beam 50;

fact parent(alice, bob).
fact parent(bob, carol).

rule ancestor(X, Y) :- parent(X, Y).
rule ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

query ancestor(alice, X)?
```

Run:

```bash
vsar run reasoning.vsar
```

## Next Steps

- **[Your First Program](first-program.md)** - Detailed tutorial
- **[Basic Queries Tutorial](../tutorials/basic-queries.md)** - Learn the basics
- **[Rules Tutorial](../tutorials/rules-and-reasoning.md)** - Deductive reasoning
- **[Examples](../examples.md)** - 6 complete programs
