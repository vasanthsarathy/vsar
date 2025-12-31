"""VSAR execution engine - orchestrates all layers."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend, MAPBackend
from vsar.language.ast import Directive, Fact, Query, Rule
from vsar.retrieval.query import Retriever
from vsar.semantics.join import initial_candidates_from_atom, join_with_atom
from vsar.semantics.substitution import Substitution, get_atom_unique_variables
from vsar.symbols.registry import SymbolRegistry
from vsar.trace.collector import TraceCollector


class QueryResult(BaseModel):
    """Result of a query execution.

    Attributes:
        query: Original query
        results: List of (entity, score) tuples
        trace_id: ID of the trace event for this query
    """

    query: Query
    results: list[tuple[str, float]]
    trace_id: str


class VSAREngine:
    """VSAR execution engine.

    Orchestrates all VSAR layers: kernel, symbols, encoding, KB, retrieval, and tracing.
    Configured via directives from VSAR programs.

    Example:
        >>> directives = [
        ...     Directive(name="model", params={"type": "FHRR", "dim": 512, "seed": 42})
        ... ]
        >>> engine = VSAREngine(directives)
        >>> engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        >>> results = engine.query(Query(predicate="parent", args=["alice", None]))
    """

    def __init__(self, directives: list[Directive]) -> None:
        """Initialize engine from directives.

        Args:
            directives: Configuration directives from VSAR program

        Raises:
            ValueError: If directives are invalid or incomplete
        """
        # Parse configuration
        self.config = self._parse_config(directives)

        # Initialize trace collector
        self.trace = TraceCollector()

        # Initialize backend
        backend_type = self.config.get("backend_type", "FHRR")
        dim = self.config.get("dim", 8192)
        seed = self.config.get("seed", 42)

        if backend_type == "FHRR":
            self.backend = FHRRBackend(dim=dim, seed=seed)
        elif backend_type == "MAP":
            self.backend = MAPBackend(dim=dim, seed=seed)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        # Initialize symbol registry
        self.registry = SymbolRegistry(self.backend, seed=seed)

        # Initialize encoder
        self.encoder = VSAEncoder(self.backend, self.registry, seed=seed)

        # Initialize KB
        self.kb = KnowledgeBase(self.backend)

        # Initialize retriever
        self.retriever = Retriever(
            self.backend,
            self.registry,
            self.kb,
            self.encoder,
        )

        # Store retrieval parameters
        self.threshold = self.config.get("threshold", 0.22)
        self.beam_width = self.config.get("beam", 50)
        self.novelty_threshold = self.config.get("novelty", 0.95)

    def _parse_config(self, directives: list[Directive]) -> dict[str, Any]:
        """Parse directives into configuration dict.

        Args:
            directives: List of directives

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If directives are invalid
        """
        config: dict[str, Any] = {}

        for directive in directives:
            if directive.name == "model":
                # @model FHRR(dim=8192, seed=42);
                backend_type = directive.params.get("type", "FHRR")
                dim = directive.params.get("dim", 8192)
                seed = directive.params.get("seed", 42)

                config["backend_type"] = backend_type
                config["dim"] = dim
                config["seed"] = seed

            elif directive.name == "threshold":
                # @threshold(value=0.22);
                config["threshold"] = directive.params.get("value", 0.22)

            elif directive.name == "beam":
                # @beam(width=50);
                config["beam"] = directive.params.get("width", 50)

            elif directive.name == "novelty":
                # @novelty(threshold=0.95);
                config["novelty"] = directive.params.get("threshold", 0.95)

        return config

    def insert_fact(self, fact: Fact) -> None:
        """Insert a fact into the knowledge base.

        Args:
            fact: Ground fact to insert

        Example:
            >>> engine.insert_fact(Fact(predicate="parent", args=["alice", "bob"]))
        """
        # Encode the fact
        atom_vec = self.encoder.encode_atom(fact.predicate, fact.args)

        # Insert into KB
        self.kb.insert(fact.predicate, atom_vec, tuple(fact.args))

    def query(self, query: Query, k: int | None = None, rules: list[Rule] | None = None) -> QueryResult:
        """Execute a query with tracing and optional rule application.

        If rules are provided, forward chaining is applied first to derive
        new facts. The query is then executed on the enriched knowledge base.

        Args:
            query: Query to execute
            k: Number of results to retrieve (default: 10)
            rules: Optional rules to apply before querying

        Returns:
            QueryResult with results and trace ID

        Example:
            >>> # Query without rules (only base facts)
            >>> result = engine.query(Query(predicate="parent", args=["alice", None]))
            >>> result.results  # [(bob, score), ...]

            >>> # Query with rules (derives facts first)
            >>> rules = [Rule(head=Atom("grandparent", ["X","Z"]),
            ...              body=[Atom("parent", ["X","Y"]), Atom("parent", ["Y","Z"])])]
            >>> result = engine.query(Query("grandparent", ["alice", None]), rules=rules)
            >>> result.results  # [(carol, score)] - derived from rules!
        """
        if k is None:
            k = 10

        # Record query start
        trace_id = self.trace.record(
            "query",
            {
                "predicate": query.predicate,
                "args": query.args,
                "variables": query.get_variables(),
                "bound_args": query.get_bound_args(),
                "has_rules": rules is not None,
            },
        )

        # Apply rules first if provided
        if rules is not None:
            from vsar.semantics.chaining import apply_rules

            chaining_result = apply_rules(self, rules, max_iterations=100, k=k)

            # Record chaining in trace
            self.trace.record(
                "chaining",
                {
                    "num_rules": len(rules),
                    "iterations": chaining_result.iterations,
                    "total_derived": chaining_result.total_derived,
                    "fixpoint_reached": chaining_result.fixpoint_reached,
                    "derived_per_iteration": chaining_result.derived_per_iteration,
                },
                parent_ids=[trace_id],
            )

        # Get variable positions and bound args
        var_positions = query.get_variables()
        bound_args = query.get_bound_args()

        if len(var_positions) != 1:
            raise ValueError(f"Query must have exactly 1 variable, got {len(var_positions)}")

        var_position = var_positions[0] + 1  # Convert to 1-indexed

        # Execute retrieval
        results = self.retriever.retrieve(
            query.predicate,
            var_position,
            bound_args,
            k=k,
        )

        # Record retrieval results
        self.trace.record(
            "retrieval",
            {
                "predicate": query.predicate,
                "var_position": var_position,
                "k": k,
                "num_results": len(results),
                "results": results[:5],  # Only store top 5 in trace
            },
            parent_ids=[trace_id],
        )

        return QueryResult(
            query=query,
            results=results,
            trace_id=trace_id,
        )

    def apply_rule(self, rule: Rule, k: int | None = None) -> int:
        """Apply a rule to derive new facts using beam search joins.

        Supports both single-body and multi-body rules. Uses beam search to
        manage combinatorial explosion in joins.

        Args:
            rule: Rule to apply
            k: Number of results to retrieve per query (default: 10)

        Returns:
            Number of derived facts added

        Example:
            >>> # Single-body: rule human(X) :- person(X).
            >>> # Multi-body: rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        """
        if k is None:
            k = 10

        if len(rule.body) == 0:
            # No body atoms - can't derive anything
            return 0

        # Check if all body predicates exist in KB
        for body_atom in rule.body:
            if not self.kb.has_predicate(body_atom.predicate):
                # Missing predicate - no derivations possible
                return 0

        # Start with first body atom
        try:
            candidates = initial_candidates_from_atom(rule.body[0], self.query, k=k, kb=self.kb)
        except ValueError:
            # First atom has unsupported structure
            return 0

        # Join with remaining body atoms
        for body_atom in rule.body[1:]:
            candidates = join_with_atom(
                candidates,
                body_atom,
                self.query,
                beam_width=self.beam_width,
                k=k,
            )

            if not candidates:
                # No candidates left after join - no derivations
                return 0

        # Apply final bindings to head and insert derived facts
        derived_count = 0
        for candidate in candidates:
            # Apply substitution to head
            ground_head = candidate.substitution.apply_to_atom(rule.head)

            # Check if head is fully ground
            if not ground_head.is_ground():
                # Head still has unbound variables - skip
                continue

            # Novelty check: avoid inserting duplicate derived facts
            atom_vec = self.encoder.encode_atom(ground_head.predicate, ground_head.args)
            if self.kb.contains_similar(
                ground_head.predicate, atom_vec, threshold=self.novelty_threshold
            ):
                # Fact already exists (similar enough) - skip
                continue

            # Insert derived fact (novel)
            self.insert_fact(Fact(predicate=ground_head.predicate, args=ground_head.args))
            derived_count += 1

        return derived_count

    def stats(self) -> dict[str, Any]:
        """Get knowledge base statistics.

        Returns:
            Dictionary with KB stats

        Example:
            >>> stats = engine.stats()
            >>> stats["total_facts"]
            100
        """
        predicates = self.kb.predicates()
        return {
            "total_facts": self.kb.count(),
            "predicates": {pred: self.kb.count(pred) for pred in predicates},
        }

    def export_kb(self, format: str = "json") -> dict[str, Any] | str:
        """Export knowledge base.

        Args:
            format: Export format ("json" or "jsonl")

        Returns:
            Exported data as dict (json) or string (jsonl)

        Raises:
            ValueError: If format is invalid

        Example:
            >>> data = engine.export_kb("json")
        """
        if format == "json":
            # Build dict representation
            data = {}
            for predicate in self.kb.predicates():
                facts = self.kb.get_facts(predicate)
                data[predicate] = [{"args": list(fact)} for fact in facts]
            return data
        elif format == "jsonl":
            # Convert to JSONL format (one fact per line)
            import json

            lines = []
            for predicate in self.kb.predicates():
                facts = self.kb.get_facts(predicate)
                for fact in facts:
                    lines.append(json.dumps({"predicate": predicate, "args": list(fact)}))
            return "\n".join(lines)
        else:
            raise ValueError(f"Invalid export format: {format}")

    def save_kb(self, path: Path | str) -> None:
        """Save knowledge base to HDF5 file.

        Args:
            path: Path to save KB

        Example:
            >>> engine.save_kb("kb.h5")
        """
        from vsar.kb.persistence import save_kb

        save_kb(self.kb, path)

    def load_kb(self, path: Path | str) -> None:
        """Load knowledge base from HDF5 file.

        Args:
            path: Path to load KB from

        Example:
            >>> engine.load_kb("kb.h5")
        """
        from vsar.kb.persistence import load_kb

        self.kb = load_kb(self.backend, path)

        # Recreate retriever with new KB
        self.retriever = Retriever(
            self.backend,
            self.registry,
            self.kb,
            self.encoder,
        )
