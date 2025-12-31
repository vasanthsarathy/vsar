# Your First VSAR Program

Step-by-step guide to creating your first VSAR program.

## Step 1: Install VSAR

```bash
pip install vsar
vsar --version
```

## Step 2: Create a Program File

Create `family.vsar`:

```prolog
// Configuration
@model FHRR(dim=1024, seed=42);

// Facts: Parent relationships
fact parent(alice, bob).
fact parent(bob, carol).

// Query: Who are alice's children?
query parent(alice, X)?
```

## Step 3: Run It

```bash
vsar run family.vsar
```

Output:
```
Inserted 2 facts

┌─────────────────────────┐
│ Query: parent(alice, X) │
├────────┬────────────────┤
│ Entity │ Score          │
├────────┼────────────────┤
│ bob    │ 0.9234         │
└────────┴────────────────┘
```

## Step 4: Add a Rule

Update `family.vsar`:

```prolog
@model FHRR(dim=1024, seed=42);
@beam(width=50);

fact parent(alice, bob).
fact parent(bob, carol).

// Rule: Derive grandparent relationship
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

query grandparent(alice, X)?
```

Run again:

```bash
vsar run family.vsar
```

Now it derives that alice is grandparent of carol!

## Step 5: Explore More

Try the example programs:

```bash
vsar run examples/03_transitive_closure.vsar
```

## Next Steps

- **[Basic Queries Tutorial](../tutorials/basic-queries.md)** - Learn querying
- **[Rules Tutorial](../tutorials/rules-and-reasoning.md)** - Learn rules
- **[Examples](../examples.md)** - 6 complete examples
