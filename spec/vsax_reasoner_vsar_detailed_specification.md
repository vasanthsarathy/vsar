# VSAX Reasoner (VSAR) — Detailed Specification

## 1. Executive summary

VSAR is a **VSA-grounded reasoning tool** built on the VSAX library. It provides a **single inference substrate** (hypervector algebra + similarity + cleanup/factorization) that can host multiple reasoning styles—starting with a practical, scalable core:

- **Deductive reasoning** over facts and Horn-style rules (Datalog-like)
- **Approximate joins** and **approximate unification** via VSA binding/unbinding and similarity search
- **Explainable results**: every answer includes a trace with similarity scores, retrieved candidates, and rule firings

VSAR is explicitly designed to address common complaints about classical Datalog/Prolog/ASP systems—**slow joins, combinatorial blowups, and brittle exact unification**—by embracing **approximate reasoning** with **massive speedups** via:

- Vectorized retrieval and batch operations (JAX/GPU)
- Bounded inference (depth/beam/threshold)
- Learned or engineered cleanup memories and optional ANN indexing

The system ships as a local-first developer tool (CLI + web UI), with optional server mode for shared KBs.

---

## 2. Project goals

### 2.1 Primary goals

1. **Scale-first reasoning**: answer queries over **very large KBs** (10^6–10^9 facts) with bounded latency using approximate inference.
2. **Unified substrate**: represent facts, rules, and explanations in a single hypervector space (per model/typing regime).
3. **Programmable**: a small reasoning language that compiles to VSA kernel ops.
4. **Explainable by construction**: traces show what was retrieved, why it matched, and which rules fired.
5. **Reproducible**: deterministic runs given seeds/config, with persisted symbol bases.

### 2.2 Non-goals (initially)

- Full classical completeness guarantees (the system prioritizes speed/approximation).
- Full ASP stable model semantics in MVP.
- Arbitrary higher-order logic.
- A “one true” probability theory. MVP uses pragmatic confidence scoring.

---

## 3. Key design principles

1. **Approximation is explicit**: every inference emits `(answer, score, trace)`.
2. **Semantics are modular**: front-end language compiles to kernel operations; new logics are additional compilers.
3. **Typed symbol spaces** reduce collisions: Entities, Relations, Properties, Contexts, etc.
4. **Bounded inference** prevents blowups: beam widths, hop limits, novelty thresholds.
5. **Two-tier execution**:
   - *Fast path*: vector retrieval (unbinding + similarity)
   - *Slow path*: factorization / resonator decoding (optional; bounded)

---

## 4. User personas and use cases

### 4.1 Personas

- **Researcher** exploring VSA reasoning paradigms.
- **Engineer** needing fast approximate querying and rule-based enrichment.
- **Analyst** running hypothesis tests (“what facts support this conclusion?”).

### 4.2 Use cases

- Knowledge graph completion and rule-based entailment.
- Rapid “semantic joins” over noisy text-extracted facts.
- Argument-like explanation traces for OSINT / intelligence analysis.
- Policy compliance checks as logical constraints (later phase).

---

## 5. MVP plan (phased)

### Phase 0 — Foundation (Weeks 1–2)

**Deliverable:** working kernel + minimal language parser + CLI.

- VSAX model selection + symbol bases + persistence
- Atom encoder + KB storage (predicate bundles)
- Retrieval query primitive with top-k results

### Phase 1 — Ground KB + conjunctive queries (Weeks 3–6)

**Deliverable:** fast approximate querying over 10^6 facts.

- Facts ingestion (CSV/JSONL/VSAR source)
- Predicate-local bundles and indexes
- Query execution: unbind → retrieve → score
- Trace output

### Phase 2 — Horn rules + bounded chaining (Weeks 7–12)

**Deliverable:** Datalog-like rules with bounded forward/backward chaining.

- Rule compilation to retrieval plans
- Beam search joins, novelty detection, semi-naive caching
- Derived fact store + provenance traces

### Phase 3 — Negation + defaults (Post-MVP)

- Classical negation as explicit atoms
- Negation-as-failure via retrieval threshold
- Prioritized defaults + exception handling

### Phase 4 — Argumentation + epistemic contexts (Post-MVP)

- Argument graphs, support/attack
- Context-indexed KBs (belief modalities)

---

## 6. Architecture

### 6.0 Compilation to kernel ops (VSA vs Clifford)

VSAR is designed so that **semantics are implemented by compilation**, not by hard-coding a specific logic engine. The compiler lowers VSARL programs into a small set of **kernel ops**. The op set is the same across modes, but the *realization* differs depending on whether the underlying algebra is VSA or Clifford.

#### 6.0.1 Intermediate representation (IR)

The compiler emits an IR graph of operations:

- `SYM(space, name)` → symbol handle
- `ENC_ATOM(pred, args[])` → atom vector
- `STORE(pred, vec)` → insert into KB bundle/index
- `RETRIEVE(space, vec, k)` → top-k nearest symbols
- `UNBIND(role_or_factor, vec)` → factor isolation primitive
- `JOIN(var, cand_sets, beam)` → approximate join with cleanup consistency
- `FIRE(rule_id, subst)` → derive head atoms
- `NOVELTY(pred, vec, θ_novel)` → membership test for insertion
- `TRACE(event, payload)` → trace emission

#### 6.0.2 Lowering differences

**VSA mode**

- `ENC_ATOM` uses role–filler binding/bundling (§8.2A)
- `UNBIND` uses approximate inverse/unbinding (model-dependent)
- `RETRIEVE` uses similarity search against basis vectors in a space

**Clifford mode**

- `ENC_ATOM` uses multivector geometric product encoding (§8.2B)
- `UNBIND` becomes algebraic factor extraction:
  - apply involution(s) (reverse/conjugate)
  - multiply by inverses/divisors of known factors
  - optionally apply grade projection before retrieval
- `RETRIEVE` is identical at the interface but operates on grade-restricted candidates

The key idea: *the same rule/query compiler emits the same IR*, but `UNBIND/ENC_ATOM` are executed by different backends.

#### 6.0.3 Example lowering snippet

Rule:

`rule grandparent(X,Z) :- parent(X,Y), parent(Y,Z).`

Compiler plan (schematic):

1. For each `X` seed (from query or from iteration frontier), build a pattern for `parent(X,Y)` and retrieve candidates for `Y`.
2. For each `Y` candidate, build a pattern for `parent(Y,Z)` and retrieve candidates for `Z`.
3. `JOIN` on `Y` cleanup consistency; aggregate scores.
4. Emit `grandparent(X,Z)` facts via `FIRE` and insert if novel.

Clifford mode uses the same plan but factor extraction yields tighter candidate sets.

---

### 6.1 High-level components

### 6.1 High-level components

1. **Language front-end**

   - Lexer/parser → AST
   - Type checker / symbol table
   - Compiler(s) → execution plan (kernel ops)

2. **VSA inference kernel (VSAX-backed)**

   - Symbol bases / typed spaces
   - Encoders for atoms/tuples/graphs
   - Store: bundled KB + indexes
   - Retrieve: similarity search, unbinding
   - Cleanup / factorization (optional)

3. **Execution engine**

   - Plan runner (vectorized)
   - Bounded search (beam)
   - Caching (memoization of subqueries)

4. **Trace & explanation layer**

   - Proof graph (nodes: facts/rule firings; edges: dependencies)
   - Scores and thresholds

5. **UI**

   - Editor + runner + results explorer
   - Trace viewer + KB inspector

### 6.2 Data flow

Program → AST → Plan → Kernel Ops → Results + Trace → UI display

---

## 7. Tech stack

### 7.1 Backend

- **Python 3.11+**
- **VSAX** as reasoning substrate (hypervectors, models, encoders, memory, factorization)
- **JAX** (already aligned with VSAX design goals) for GPU/CPU vectorization
- **FastAPI** for server mode and UI API
- **SQLite / DuckDB** (optional) for metadata and persistence of non-vector artifacts
- Optional ANN index:
  - **FAISS** (CPU/GPU) or **hnswlib** for large-scale nearest neighbors

### 7.2 Frontend

- **React + TypeScript**
- **Monaco Editor** for the language editor
- **Tailwind** for precise spacing control (tight UI)
- **IBM Plex Mono** (aka “IBM Plex Mono”) for typography
- **Black/white theme** with semantic colors only for alerts

### 7.3 Packaging

- CLI via **Typer**
- Python packaging via uv/pyproject
- Dockerfile for reproducible runs

---

## 8. Knowledge representation in VSA and Clifford Algebra

VSAX supports not only classical VSA operations (binding, bundling, similarity) but also **Clifford algebra–inspired operators** (e.g., geometric product–like compositions, multivector representations, and involutions). These capabilities are *directly relevant* to VSAR and expand the expressivity and robustness of the reasoning substrate.

### 8.1 Typed symbol spaces

Each space has its own basis set to reduce collisions and control algebraic behavior:

- `E` Entities
- `R` Relations (predicates)
- `A` Attributes / literals
- `C` Contexts (beliefs, agents)
- `T` Time
- `S` Structural / operator symbols (used for Clifford-style composition)

Symbol spaces may be implemented either as:

- **Pure VSA spaces** (e.g., FHRR, MAP)
- **Clifford-enriched spaces**, where symbols correspond to blades or graded subspaces

The choice is per-model and declared via directives (see §9). Each space has its own basis set to reduce collisions:

- `E` Entities
- `R` Relations (predicates)
- `A` Attributes / literals
- `C` Contexts (optional)
- `T` Time (optional)

### 8.2 Atom encoding

We support **two compatible encoding regimes**:

#### (A) Classical VSA role–filler encoding

As before, a k-ary atom `p(t1,…,tk)` is encoded using role–filler binding and bundling.

Let roles be `ρ1,…,ρk`. Let `⊗` denote VSA binding and `⊕` bundling.

```
enc_VSA(p(t1,…,tk)) = hv(p) ⊗ ( (hv(ρ1) ⊗ hv(t1)) ⊕ … ⊕ (hv(ρk) ⊗ hv(tk)) )
```

#### (B) Clifford-enriched encoding (optional)

When Clifford operators are enabled, atoms may be encoded as **multivectors**:

- Predicates correspond to basis blades (e.g., grade-1 vectors)
- Roles correspond to orthogonal blades
- Fillers correspond to blades or vectors in typed subspaces

Using a geometric-product–like operation `⋆`:

```
enc_Cliff(p(t1,…,tk)) = hv(p) ⋆ hv(ρ1) ⋆ hv(t1) ⋆ … ⋆ hv(ρk) ⋆ hv(tk)
```

This representation preserves **order and orientation information** and enables additional algebraic queries (e.g., involution, reversal, grade projection).

Both encodings map to the same similarity-based retrieval interface. We represent a k-ary atom `p(t1,…,tk)` as a role-filler structure.

Let roles be `ρ1,…,ρk` (fixed per arity). Let `hv(x)` be a hypervector for symbol `x`. Let `⊗` be binding and `⊕` be bundling/superposition.

**Atom encoding:**

`enc(p(t1,…,tk)) = hv(p) ⊗ ( (hv(ρ1) ⊗ hv(t1)) ⊕ … ⊕ (hv(ρk) ⊗ hv(tk)) )`

Optionally include type tags:

`hv(ti) := hv(type(ti)) ⊗ hv(id(ti))`

### 8.3 KB storage

For each predicate `p`, maintain a bundled store:

`KB[p] = ⊕_{f ∈ Facts(p)} enc(f)`

Optionally maintain a separate index over fact vectors for retrieval.

### 8.4 Query patterns

Query compilation depends on the chosen algebra.

#### VSA mode

Queries are handled via approximate unbinding and similarity, as described previously.

#### Clifford mode

Queries may additionally use:

- **Left/right geometric division** to isolate factors
- **Grade projection** to restrict candidate extraction to specific structural components
- **Involutions** (reverse, conjugate) to test symmetry or order sensitivity

Example: for a binary predicate encoded as `p ⋆ ρ1 ⋆ X ⋆ ρ2 ⋆ bob`, the candidate for `X` may be extracted by:

```
X̂ = grade_1( reverse(ρ1) ⋆ reverse(p) ⋆ q )
```

followed by cleanup in entity space `E`.

This yields more stable factorization when role order or nesting depth matters. A query may have variables. We compile variables into **placeholders** that are resolved by unbinding + similarity.

Example: query `parent(X, bob)`

Construct pattern:

`q = hv(parent) ⊗ ( (hv(ρ1) ⊗ VAR_X) ⊕ (hv(ρ2) ⊗ hv(bob)) )`

To retrieve candidates for `X`, unbind `hv(ρ1)` and remove predicate tag:

`cand_X = unbind( unbind(q, hv(parent)), hv(ρ1) )`

Then retrieve nearest neighbors of `cand_X` in entity space `E`.

---

## 9. Reasoning language (VSARL)

### 9.1 Design goals

- Familiar to Datalog/Prolog users
- Strongly typed (optional annotations)
- Explicit approximation controls (thresholds, beam sizes)
- Trace is first-class

### 9.2 Lexical structure

- Comments: `// ...` and `/* ... */`
- Identifiers:
  - Predicates: lower\_snake `parent`
  - Constants: `bob`, `tufts`
  - Variables: UpperCamel `X`, `Person`

### 9.3 Core syntax

#### Facts

```
fact parent(alice, bob).
fact lives_in(bob, boston).
```

#### Rules (Horn)

```
rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

#### Classical negation

```
fact !enemy(alice, bob).
```

#### Negation-as-failure (default negation)

```
rule safe(X) :- person(X), not enemy(X, _).
```

#### Query

```
query grandparent(alice, Z)?
```

#### Directives

```
@model FHRR(dim=8192, seed=1);
@threshold 0.22;
@beam 50;
@max_hops 3;
@trace full;
```

### 9.4 Extended constructs (post-MVP)

- Context blocks:

```
context alice {
  fact believes(rainy(today)).
}
```

- Argument blocks:

```
argument a1: claim p. support q, r. attack a2.
```

---

## 10. Formal semantics (MVP, VSA + Clifford)

### 10.1 Domains and symbols

Let `Σ_E, Σ_R, Σ_S` be disjoint symbol sets for entities, predicates, and structural operators.

Let `V` be a representation space that may be:

- a classical VSA space, or
- a Clifford algebra–valued multivector space

The space supports:

- composition (`⊗` for VSA, `⋆` for Clifford)
- (approximate) inverse or division
- similarity `sim : V×V→[0,1]`
- cleanup `clean_S : V→Σ_S`

When Clifford mode is enabled, `V` is graded and supports projections `π_k` onto grade-k subspaces. Let `Σ_E, Σ_R` be disjoint symbol sets for entities and predicates. Let `V` be the hypervector space for a chosen VSA model with operations:

- bundling `⊕ : V×V→V`
- binding `⊗ : V×V→V`
- (approx.) unbinding `⊘ : V×V→V` (model-dependent)
- similarity `sim : V×V→[0,1]`
- cleanup `clean_S : V→Σ` mapping vectors to nearest symbol in space `S` (entity/predicate space)

Let `hv_S : Σ_S → V` map symbols to basis hypervectors (per space `S`).

### 10.2 Interpretation function

An interpretation `I` comprises:

- For each predicate `p ∈ Σ_R`, a bundled store `KB[p] ∈ V`
- For each symbol space `S`, a cleanup memory containing basis vectors `{hv_S(s)}`

### 10.3 Approximate satisfaction of ground atoms

For a ground atom `a = p(t1,…,tk)` define its encoding `enc(a)` as in §8.

Define membership score:

`μ_I(a) := sim(enc(a), KB[p])`

A ground atom is **accepted** at threshold `θ` iff:

`I ⊨_θ a  ⇔  μ_I(a) ≥ θ`

### 10.4 Query answering (single atom with one variable)

Query answering is parameterized by algebra.

#### VSA semantics

As previously defined: unbinding + similarity + cleanup.

#### Clifford semantics

Given a query multivector `q`, variable isolation is defined via algebraic factor extraction:

1. Apply appropriate involution(s) to normalize orientation.
2. Multiply by inverses of known components.
3. Project onto the expected grade.

The resulting vector is cleaned up in the relevant symbol space.

This supports more stable decoding for deeply nested or ordered structures. For query `p(X, c)` with variable in position 1:

1. Construct pattern vector `q` with placeholder in role 1.
2. Compute candidate vector:

`v_X := (q ⊘ hv_R(p)) ⊘ hv(ρ1)`

3. Return top-k entities `e` ranked by `sim(v_X, hv_E(e))`.

### 10.5 Conjunctive query semantics (bounded)

A conjunctive query `Q = a1 ∧ … ∧ an` is answered by a bounded search procedure:

- For each atom `ai`, compute candidate sets for its variables by retrieval.
- Combine candidate sets via approximate joins using:
  - shared variable agreement by cleanup consistency
  - score aggregation (e.g., product or min)

Let `Ans(Q)` be the set of substitutions `σ` found by the procedure. Each answer has a score `Score(σ)` computed from retrieval similarities.

This is **procedural semantics** (intentionally), because approximation + bounds define meaning.

### 10.6 Rule application (Horn rules)

A rule `h :- b1,…,bn` is applied by:

1. Evaluate body conjunctive query `b1∧…∧bn` to obtain substitutions `{σ}`.
2. For each σ, derive ground head `hσ`.
3. Insert `enc(hσ)` into `KB[head_pred]` if novel:

Novelty test: `sim(enc(hσ), KB[head_pred]) < θ_novel`

Repeat until fixpoint or max iterations.

### 10.7 Negation

- **Classical negation** `!p(t)` is a separate predicate symbol (or a signed predicate) stored explicitly.
- **Default negation** `not p(t)` succeeds when `μ_I(p(t)) < θ`.

In MVP, negation is **stratified**: no cycles through `not`.

### 10.8 Explanation trace semantics

Every retrieval and rule firing produces a trace node with:

- kernel op id
- retrieved candidates + similarities
- chosen substitutions + score aggregation
- derived facts + novelty decision

The trace is a DAG; answers reference a subgraph.

---

## 11. Why Clifford? (and when to use it)

This section explains why VSAX’s Clifford-inspired operators matter for VSAR, and when you should enable them.

### 11.1 Bundling vs. geometric product (intuition)

Classical VSA reasoning typically relies on:

- **Bundling (superposition)** to represent sets/multisets of components
- **Binding/unbinding** to attach roles to fillers and later retrieve them approximately

This works well, but two pain points arise in large structured KBs:

1. **Order/orientation loss**: bundling is commutative; many encodings become insensitive to the order of components.
2. **Factorization fragility**: deeply nested structures can produce ambiguous or noisy unbinding targets.

Clifford-enriched representations address this by introducing a **geometric-product–like composition** that is:

- **Associative** and often **order-sensitive**
- Supports **involutions** (reverse, conjugate) and **grade projections** that can isolate structural parts

### 11.2 Practical benefits for reasoning

Enable Clifford mode when you want:

- **Order-sensitive structures** (sequences, nested terms, paths)
- **More stable factorization** for multi-hop explanations
- **Structural invariants** (symmetry/orientation tests) in traces
- **Typed decomposition** via grade projections (reduce collisions)

Stick with pure VSA mode when you want:

- Maximum throughput and simplicity
- Mainly set-like relational facts with shallow arity

### 11.3 Side-by-side comparison (conceptual)

- **Bundling**: `a ⊕ b = b ⊕ a` (great for sets; order is erased)
- **Geometric product**: `a ⋆ b ≠ b ⋆ a` (captures orientation/order; better for structured terms)

---

## 12. Performance strategy (addressing Datalog/ASP blowups)

### 11.1 Why classical systems blow up

Classical Datalog/Prolog/ASP often incur:

- expensive joins over large relations
- combinatorial unification over variable bindings
- exponential search in worst cases (especially with negation/disjunction)

### 11.2 VSAR’s approach

VSAR tackles this head-on with:

1. **Approximate joins by vector retrieval**

   - Replace relational join enumeration with similarity search.

2. **Beam search + thresholds**

   - Only the top-k candidates per variable are explored.

3. **Vectorized execution**

   - Batch compute similarities; use GPU when available.

4. **Predicate partitioning**

   - Separate bundles per predicate reduce noise and speed search.

5. **Caching and semi-naive rule evaluation**

   - Memoize subquery results.

6. **Novelty detection**

   - Avoid reinserting near-duplicate derived facts.

This yields large speedups at the cost of completeness, but the trade is **explicit and tunable**.

---

## 12. UI/UX design specification

### 12.1 Visual language

- **Monochrome**: black/white only
- **Accent colors reserved** for:
  - Errors (red)
  - Warnings (amber)
  - Success/OK (green)
  - Active selection (blue)
- Typography: **IBM Plex Mono** everywhere
- Shapes: **sharp corners**, no rounding
- Spacing: **tight**; minimal padding (2–6px)
- Borders: 1px high-contrast lines

### 12.2 Layout

A three-panel “IDE-like” grid:

1. **Left sidebar (KB + Runs)**

   - KB browser: predicates, counts, last updated
   - Symbol spaces: entity count, dim, model
   - Run history: timestamp, program name, status

2. **Center (Editor + Console)**

   - Monaco editor with VSARL syntax highlighting
   - Toolbar: Run, Stop, Clear, Save, Load
   - Console output: logs, timings, warnings

3. **Right (Results + Trace)**

   - Query results table: bindings + scores
   - Trace viewer: expandable DAG steps
   - Fact inspector: click a fact → show encoding components and nearest neighbors

### 12.3 Key interactions

- **Run program**: executes all directives + facts + rules + queries in file order.
- **Inline diagnostics**: parse/type errors highlighted in editor.
- **Explain mode**: click an answer to open its trace subtree.
- **Threshold tuning**: a compact control strip to adjust θ, beam, hops and rerun.

### 12.4 Information architecture

- Top bar: project name + model info + runtime target (CPU/GPU)
- Status line: last run latency, facts loaded, derived facts, cache hit rate

### 12.5 Accessibility

- High contrast by default
- Keyboard-first workflow (Ctrl+Enter run)

---

## 13. Distribution, developer experience, and packaging

VSAR is designed as a **three-layer system**:

1. **Formal core** (language + semantics)
2. **Python library** (parser, compiler, kernel, APIs)
3. **Interactive web playground** (demo, learning, exploration)

This section specifies how these layers fit together.

---

### 13.1 Python library (`vsar`)

VSAR ships as a standard Python package:

```
pip install vsar
```

The library exposes:

#### 13.1.1 Public modules

```
vsar/
  language/        # grammar, AST, parser
  semantics/       # formal semantics, execution engine
  kernel/          # VSA / Clifford backends (VSAX-based)
  kb/              # KB storage, indexing, novelty detection
  trace/           # trace graph structures
  cli/             # CLI entry points
  utils/
```

#### 13.1.2 Core Python API

```python
from vsar import Program, Engine

prog = Program.from_file("example.vsar")
engine = Engine(model="FHRR", dim=8192)

result = engine.run(prog)
print(result.answers)
print(result.trace)
```

The API allows:

- program loading/parsing
- incremental fact ingestion
- rule execution
- query evaluation
- access to traces, scores, and derived facts

This enables **embedding VSAR inside other Python systems**.

---

### 13.2 Command-line interface (CLI)

The CLI is a thin wrapper over the Python API and mirrors the web UI functionality.

#### 13.2.1 Commands

```
vsar run program.vsar
vsar repl
vsar ingest facts.jsonl --predicate parent
vsar export --format json
vsar inspect kb
vsar serve --port 8080
```

#### 13.2.2 CLI design principles

- Scriptable and composable (Unix-style)
- Deterministic runs
- Machine-readable outputs (JSON)
- Identical semantics to the library and web UI

---

### 13.3 Web UI & playground

The Web UI serves **three roles**:

1. **Interactive playground** (like clingo’s ASP playground)
2. **Demo environment** for talks and tutorials
3. **Visual debugger** for reasoning and traces

#### 13.3.1 Architecture

- Frontend: React + TypeScript
- Backend: FastAPI using the same `vsar.Engine`
- Stateless execution per run (unless explicitly saved)

No special "UI-only" semantics exist — the UI is a client of the same engine.

#### 13.3.2 Playground features

- In-browser VSARL editor (Monaco)
- Example programs gallery
- One-click run / reset
- Adjustable thresholds, beam widths, hop limits
- Results panel with sortable answers
- Trace explorer (expand/collapse reasoning DAG)

#### 13.3.3 Demo mode

A curated set of examples:

- Datalog-style inference
- Approximate joins over noisy facts
- Clifford vs VSA comparison toggles
- Performance scaling demos

---

### 13.4 Documentation & learning resources

VSAR includes **first-class documentation** as part of the project, not an afterthought.

#### 13.4.1 Documentation site

A static documentation site (e.g., MkDocs or Docusaurus) containing:

- **Overview**: philosophy, approximate reasoning, VSA vs classical logic
- **Language reference**: full VSARL syntax
- **Formal semantics**: readable version of §10
- **Tutorials**: step-by-step reasoning examples
- **User guide**: common patterns, performance tuning
- **API reference**: Python classes and methods

#### 13.4.2 Learning path

Suggested learning progression:

1. Hello world: facts + queries
2. Conjunctive queries
3. Rules and chaining
4. Approximation controls
5. Traces and explanations
6. Clifford mode (advanced)

Each step includes runnable code and playground links.

#### 13.4.3 Literate examples

Examples are maintained in:

- `.vsar` files
- Jupyter/Marimo notebooks

Each example cross-links:

- source program
- explanation text
- expected output

---

### 13.5 Reproducibility & versioning

- Language versions are explicit (e.g., `@lang 0.1`)
- Model configs are serialized
- Symbol bases are persisted and reloadable
- Playground runs can be exported as `.vsar` bundles

---

### 13.6 Community & extensibility (future)

- Plugin system for new semantics compilers
- Custom backends (e.g., neuromorphic / hardware VSA)
- External KB connectors

---

## 14. Software engineering, quality, and release practices

VSAR is intended to be a **serious, long-lived research and engineering system**, not a throwaway prototype. As such, strong software engineering practices are considered part of the core specification.

### 14.1 Repository structure & version control

- GitHub-hosted repository (canonical)
- Main branches:
  - `main`: stable, releasable code
  - `dev`: active development
- Feature branches: `feature/*`, `fix/*`
- All changes via pull requests (even for single-developer workflow)

Commit practices:

- Small, atomic commits
- Descriptive commit messages
- One logical change per commit

---

### 14.2 Code quality & linting

Mandatory automated checks:

- **Formatting**: `black` (Python), `prettier` (TS/JS)
- **Linting**: `ruff` (Python), `eslint` (frontend)
- **Type checking**: `mypy` (Python), TypeScript strict mode

Pre-commit hooks (required):

- Code formatting
- Lint checks
- Static type checks
- Basic security checks

No code may be merged without passing pre-commit checks.

---

### 14.3 Testing strategy

Testing is first-class and extensive.

#### 14.3.1 Test types

- **Unit tests**

  - Parser, AST, compiler
  - Kernel ops (VSA + Clifford backends)
  - Similarity, cleanup, novelty detection

- **Integration tests**

  - End-to-end program execution
  - CLI invocation tests
  - API tests (server mode)

- **Semantic tests**

  - Small KBs with known closure
  - Approximate invariants (monotonicity under added facts, etc.)

- **Regression tests**

  - Fixed seeds for reproducibility

#### 14.3.2 Coverage requirements

- Target ≥ **90% line coverage** for core modules
- Critical logic paths must have explicit tests
- Coverage enforced via CI

---

### 14.4 Continuous integration (CI)

GitHub Actions–based CI pipeline:

On every PR and push to `main`:

- Install dependencies
- Run pre-commit hooks
- Run full test suite
- Enforce coverage thresholds
- Build documentation

CI failures block merges.

---

### 14.5 Continuous delivery & releases

#### 14.5.1 Versioning

- Semantic versioning: `MAJOR.MINOR.PATCH`
- Language versioning independent but tracked (e.g., `@lang 0.1`)

Version bumps required for:

- Language changes
- Semantics changes
- Kernel changes

#### 14.5.2 Release process

1. Merge to `main`
2. Bump version
3. Update `CHANGELOG.md`
4. Tag release (`vX.Y.Z`)
5. Build artifacts
6. Publish to PyPI

Releases are automated via GitHub Actions.

---

### 14.6 PyPI publishing

- Package name: `vsar`
- Wheel + source distributions
- Verified builds
- Reproducible environments

Users install via:

```
pip install vsar
```

---

### 14.7 Documentation & README quality

Documentation quality is treated as a core deliverable.

#### 14.7.1 README.md (professional-grade)

Must include:

- Clear project description and motivation
- Installation instructions
- Quickstart example
- Links to playground and docs
- Citation / attribution

#### 14.7.2 Documentation site

- Automatically built and deployed
- Versioned with releases
- Includes tutorials, API reference, and language spec

---

### 14.8 Documentation testing

- Code examples in docs are executable
- CI runs doctests / notebook checks
- Playground examples mirror documentation examples

---

### 14.9 Long-term maintainability

- Clear deprecation policies
- Backward compatibility notes
- Migration guides when semantics change

---

## 15. APIs (server mode)

### 13.1 Commands

- `vsar run program.vsar`
- `vsar repl`
- `vsar ingest facts.jsonl --predicate parent`
- `vsar export --format json`
- `vsar serve --port 8080`

### 13.2 Output formats

- Human table output
- JSON with trace references

---

## 14. APIs (server mode)

- `POST /run` → executes a program, returns results + traces
- `GET /kb/predicates` → list predicates
- `GET /kb/predicate/{p}` → counts, sample, stats
- `GET /trace/{id}` → trace subgraph

---

## 15. Evaluation & benchmarks (MVP)

### 15.1 Metrics

- Latency per query (p50/p95)
- Recall\@k on held-out KG completion tasks
- Trace size and interpretability
- Derived fact growth and convergence behavior

### 15.2 Datasets

- Synthetic rulesets with known closure
- A medium KG dataset (e.g., family relations + geography)

---

## 16. Security & privacy

- Local-first by default
- Server mode: optional auth token
- No external calls required

---

## 17. Compatibility with structured logics (Description Logics, OWL, etc.)

## 17a. Relationship to LLM semantic knowledge (scope and boundaries)

VSAR is intentionally designed to be **compatible with, but not dependent on, Large Language Models (LLMs)**. This section clarifies scope and future integration points.

### 17a.1 Motivation

Logical symbols such as `grandparent`, `enemy`, or `causes` are not merely syntactic tokens: they correspond to **shared world concepts** that humans (and LLMs) already model richly. LLMs encode:

- lexical similarity and synonymy (e.g., misspellings, paraphrases)
- typical role structures and argument expectations
- commonsense constraints and defaults

VSAR’s approximate, similarity-based reasoning substrate naturally aligns with these properties.

### 17a.2 Design stance (important)

VSAR **does not delegate inference to an LLM** and does **not require an LLM to be present**.

Instead:

> **LLMs are treated as optional semantic priors that can help ground symbols and improve robustness, not as reasoning engines.**

All reasoning remains within the VSAR algebraic kernel.

### 17a.3 Where LLM integration fits (in-scope)

The following are explicitly in-scope for VSAR *as optional extensions*:

1. **Symbol grounding assistance**

   - Use an LLM to map surface strings → canonical symbols
   - Handle misspellings, paraphrases, or variant names
   - Example: `granparent` → `grandparent`

2. **Similarity biasing**

   - Initialize or adjust symbol hypervectors using LLM embedding similarity
   - Encourage related predicates/concepts to be closer in VSA space

3. **Type and arity hints**

   - Use LLMs to suggest likely argument structures
   - Enforce soft constraints during rule compilation

4. **Explanation alignment**

   - Translate VSAR traces into natural-language explanations using an LLM
   - (Purely presentational; no semantic authority)

### 17a.4 Where LLMs are explicitly *out of scope*

- LLMs deciding logical truth or rule firing
- LLMs replacing the inference engine
- Prompt-based reasoning in place of kernel execution

VSAR remains:

- deterministic (given seeds)
- inspectable
- algebraically grounded

### 17a.5 Why this matters for robustness

This separation yields a powerful hybrid:

- **VSAR**: structure, constraints, bounded inference, traceability
- **LLMs**: lexical robustness, commonsense priors, noise tolerance

This makes VSAR resilient to:

- misspellings
- synonym variation
- partially specified or noisy inputs

without sacrificing control or explainability.

### 17a.6 Future roadmap (optional)

- LLM-assisted symbol resolution module
- Hybrid benchmarks: noisy inputs vs classical reasoners
- Evaluation of retrieval robustness under lexical perturbations

---

VSAR is explicitly designed to remain compatible with **more structured logical formalisms**, including **Description Logics (DL)** and OWL-style knowledge representation, even though these are *not* part of the MVP.

### 17.1 Why DL compatibility matters

Description Logics introduce:

- Explicit **concept hierarchies** (TBoxes)
- **Role restrictions** (∃R.C, ∀R.C)
- **Cardinality constraints**
- Clear separation between **terminological knowledge (TBox)** and **assertional knowledge (ABox)**

These features are important for ontology-driven reasoning, semantic web systems, and knowledge engineering.

### 17.2 VSAR as a compilation target for DL

VSAR does **not** aim to re-implement classical DL tableaux algorithms. Instead:

> **DLs are treated as high-level specification languages that can be compiled into VSAR’s algebraic inference substrate.**

Concretely:

- **Concepts** → unary predicates / typed entity spaces
- **Roles** → binary predicates with structural constraints
- **Subsumption (C ⊑ D)** → rules or structural similarity constraints
- **Existential restrictions (∃R.C)** → compiled into bounded retrieval + witness construction
- **Universal restrictions (∀R.C)** → compiled into constraint checks over retrieved neighbors

This mirrors how VSAR treats Datalog, defaults, and argumentation: *as front-end languages that lower into the same kernel ops*.

### 17.3 TBox / ABox separation

VSAR can naturally support a DL-style separation:

- **TBox**: stored as rules, constraints, and similarity biases
- **ABox**: stored as ground facts in KB bundles

This allows:

- Fast approximate ABox reasoning
- Slower, more structured TBox-driven constraint propagation

### 17.4 Approximate DL semantics

Classical DL reasoning is exact and often worst-case exponential. VSAR instead supports:

- **Approximate subsumption** (similarity-based concept membership)
- **Soft constraints** instead of hard contradictions
- **Bounded model construction** for existential restrictions

This makes VSAR suitable for:

- Noisy ontologies
- Learned concept embeddings
- Hybrid symbolic–statistical systems

### 17.5 Future roadmap items

Planned DL-related extensions include:

- DL-lite / EL++ style fragment compiler
- OWL import (restricted profiles)
- Concept lattice visualization in the UI
- Mixed DL + rule reasoning

Importantly, *no changes to the VSAR kernel are required* to support these — only additional front-end compilers and constraints.

---

## 18. Roadmap beyond MVP

- Stable-model / ASP-like compilation (bounded search)
- Probabilistic overlays (calibration, ensembles)
- Argumentation semantics module
- Epistemic contexts and belief revision
- Visual query builder

---

## 18. Worked example: where Clifford helps

This example illustrates a case where a pure VSA encoding can become ambiguous (order/role confusion under heavy superposition), and how Clifford mode can improve stability.

### 18.1 Setup

Facts:

- `edge(a,b)`, `edge(b,c)`, `edge(c,d)`
- `edge(a,c)` (a shortcut)

Goal query:

- Find nodes `Z` such that there exists a length-2 path `path2(a, Z)` where:
  - `path2(X,Z) :- edge(X,Y), edge(Y,Z)`

### 18.2 Pure VSA behavior (where it can wobble)

In VSA mode, `edge(X,Y)` retrieval often proceeds by unbinding a bundled store `KB[edge]`.

When the KB is large and `edge` is heavily bundled, extracting a clean candidate for `Y` from `edge(a,Y)` and then joining against `edge(Y,Z)` can degrade:

- `Ŷ` is a noisy vector close to multiple entity vectors
- beam search may include several plausible `Y` candidates
- the join becomes approximate and may admit spurious `Z`

This is not a bug: it is the intended tradeoff of approximate joins.

### 18.3 Clifford mode (improving factor isolation)

In Clifford mode, represent a binary edge as an ordered multivector product:

`enc(edge(u,v)) = hv(edge) ⋆ hv(ρ1) ⋆ hv(u) ⋆ hv(ρ2) ⋆ hv(v)`

To retrieve `Y` from `edge(a,Y)`, the engine can:

- multiply by inverses of the known left factors (predicate + role + `a`)
- use an involution (e.g., reverse) to normalize orientation
- project onto the expected grade (entity grade)

This tends to produce a cleaner `Ŷ` in the entity subspace, reducing beam width needed and lowering spurious joins.

### 18.4 What the trace shows

In explain mode, the trace can show:

- VSA mode: larger candidate sets for `Y`, lower similarity margins
- Clifford mode: fewer candidates, higher top-1 margin, more stable `Y` cleanup

---

## Appendix A: Example program

```
@model FHRR(dim=8192, seed=1);
@threshold 0.20;
@beam 50;
@max_hops 3;
@trace full;

fact parent(alice, bob).
fact parent(bob, carol).

rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

query grandparent(alice, Z)?
```

Expected result: `Z=carol` with score and a trace showing retrieval of `Y=bob` and firing the rule.

