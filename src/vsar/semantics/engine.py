"""VSAR execution engine - orchestrates all layers."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from vsar.encoding.roles import RoleVectorManager
from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend, MAPBackend
from vsar.language.ast import Directive, Fact, Query
from vsar.retrieval.query import Retriever
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

        # Initialize role manager
        self.role_manager = RoleVectorManager(self.backend, seed=seed)

        # Initialize retriever
        self.retriever = Retriever(
            self.backend,
            self.registry,
            self.kb,
            self.encoder,
            self.role_manager,
        )

        # Store retrieval parameters
        self.threshold = self.config.get("threshold", 0.22)
        self.beam_width = self.config.get("beam", 50)

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

    def query(self, query: Query, k: int | None = None) -> QueryResult:
        """Execute a query with tracing.

        Args:
            query: Query to execute
            k: Number of results to retrieve (default: 10)

        Returns:
            QueryResult with results and trace ID

        Example:
            >>> result = engine.query(Query(predicate="parent", args=["alice", None]))
            >>> result.results  # [(entity, score), ...]
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
            },
        )

        # Get variable positions and bound args
        var_positions = query.get_variables()
        bound_args = query.get_bound_args()

        if len(var_positions) != 1:
            raise ValueError(
                f"Query must have exactly 1 variable, got {len(var_positions)}"
            )

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
                    lines.append(
                        json.dumps({"predicate": predicate, "args": list(fact)})
                    )
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
            self.role_manager,
        )
