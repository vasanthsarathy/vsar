"""Query engine for VSAR - answering queries via unbind→cleanup."""

from dataclasses import dataclass
from typing import Optional

from ..unification.decoder import StructureDecoder, Atom, Constant, Variable
from ..store.fact_store import FactStore


@dataclass
class QueryResult:
    """Result of a query with bindings and score."""
    bindings: dict[str, str]  # Variable name -> value
    score: float

    def __repr__(self) -> str:
        """String representation."""
        bindings_str = ", ".join(f"{k}={v}" for k, v in self.bindings.items())
        return f"QueryResult({{{bindings_str}}}, score={self.score:.2f})"


class QueryEngine:
    """
    Query answering engine using unbind→cleanup.

    Answers queries like parent(alice, ?X) by:
    1. Retrieving candidate facts by predicate
    2. Decoding each fact
    3. Verifying bound arguments match
    4. Extracting variable bindings
    5. Returning results with scores

    Args:
        decoder: Structure decoder for decoding facts
        fact_store: Fact store for retrieval

    Example:
        >>> engine = QueryEngine(decoder, fact_store)
        >>> query = Atom("parent", [Constant("alice"), Variable("X")])
        >>> results = engine.answer_query(query, threshold=0.1)
        >>> for result in results:
        ...     print(f"X = {result.bindings['X']}, score = {result.score}")
    """

    def __init__(self, decoder: StructureDecoder, fact_store: FactStore):
        """Initialize the query engine.

        Args:
            decoder: Structure decoder
            fact_store: Fact store
        """
        self.decoder = decoder
        self.fact_store = fact_store

    def answer_query(
        self,
        query: Atom,
        threshold: float = 0.1,
        max_results: int = 100
    ) -> list[QueryResult]:
        """
        Answer a query and return variable bindings.

        Args:
            query: Query atom with Variables for unknown positions
            threshold: Minimum decoding threshold
            max_results: Maximum number of results to return

        Returns:
            List of QueryResult with variable bindings and scores

        Example:
            >>> query = Atom("parent", [Constant("alice"), Variable("X")])
            >>> results = engine.answer_query(query)
            >>> results[0].bindings  # {'X': 'bob'}
        """
        # Get variable positions
        var_positions = {}
        for i, arg in enumerate(query.args):
            if isinstance(arg, Variable):
                var_positions[i] = arg.name

        if not var_positions:
            # Ground query - just check if fact exists
            return self._answer_ground_query(query, threshold)

        # Retrieve candidate facts by predicate
        candidates = self.fact_store.retrieve_by_predicate(query.predicate)

        results = []
        for fact_item in candidates:
            # Decode fact
            decoded = self.decoder.decode_atom(fact_item.vec, threshold=threshold)
            if decoded is None:
                continue

            # Check predicate matches
            if decoded.predicate != query.predicate:
                continue

            # Check arity matches
            if len(decoded.args) != len(query.args):
                continue

            # Check bound arguments match
            match = True
            for i, query_arg in enumerate(query.args):
                if isinstance(query_arg, Constant):
                    # Bound argument - must match
                    if i >= len(decoded.args):
                        match = False
                        break
                    decoded_arg = decoded.args[i]
                    if not isinstance(decoded_arg, Constant):
                        match = False
                        break
                    if decoded_arg.name != query_arg.name:
                        match = False
                        break

            if not match:
                continue

            # Extract variable bindings
            bindings = {}
            for pos, var_name in var_positions.items():
                if pos < len(decoded.args):
                    decoded_arg = decoded.args[pos]
                    if isinstance(decoded_arg, Constant):
                        bindings[var_name] = decoded_arg.name
                    else:
                        # Couldn't decode this position
                        match = False
                        break

            if not match or len(bindings) != len(var_positions):
                continue

            # Create result with score from item weight
            result = QueryResult(bindings=bindings, score=fact_item.weight)
            results.append(result)

            if len(results) >= max_results:
                break

        # Sort by score (descending)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _answer_ground_query(
        self,
        query: Atom,
        threshold: float
    ) -> list[QueryResult]:
        """
        Answer a ground query (no variables).

        Returns empty list if not found, or list with single empty binding
        if found.

        Args:
            query: Ground query atom
            threshold: Decoding threshold

        Returns:
            List with single QueryResult if found, empty list otherwise
        """
        candidates = self.fact_store.retrieve_by_predicate(query.predicate)

        for fact_item in candidates:
            decoded = self.decoder.decode_atom(fact_item.vec, threshold=threshold)
            if decoded is None:
                continue

            # Check if all arguments match
            if decoded.predicate != query.predicate:
                continue
            if len(decoded.args) != len(query.args):
                continue

            match = True
            for q_arg, d_arg in zip(query.args, decoded.args):
                if not isinstance(q_arg, Constant) or not isinstance(d_arg, Constant):
                    match = False
                    break
                if q_arg.name != d_arg.name:
                    match = False
                    break

            if match:
                return [QueryResult(bindings={}, score=fact_item.weight)]

        return []

    def __repr__(self) -> str:
        """String representation."""
        return f"QueryEngine({len(self.fact_store)} facts)"
