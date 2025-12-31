"""Join operations for multi-body rule execution with beam search."""

from typing import Any

from pydantic import BaseModel, Field

from vsar.language.ast import Atom, Query
from vsar.semantics.substitution import Substitution, get_atom_unique_variables


class CandidateBinding(BaseModel):
    """A candidate variable binding with a score.

    Represents a partial binding of variables to values discovered during
    rule execution, along with a confidence score.

    Attributes:
        substitution: Variable → value mappings
        score: Confidence score (product of all contributing query scores)

    Example:
        >>> binding = CandidateBinding(
        ...     substitution=Substitution(bindings={"X": "alice", "Y": "bob"}),
        ...     score=0.85
        ... )
    """

    substitution: Substitution = Field(..., description="Variable bindings")
    score: float = Field(..., description="Confidence score (0.0 to 1.0)")

    def extend(self, var: str, value: str, score: float) -> "CandidateBinding":
        """Extend this binding with a new variable.

        Args:
            var: Variable name
            value: Value to bind
            score: Score contribution from this binding

        Returns:
            New CandidateBinding with extended substitution and updated score

        Example:
            >>> binding = CandidateBinding(
            ...     substitution=Substitution(bindings={"X": "alice"}),
            ...     score=0.9
            ... )
            >>> extended = binding.extend("Y", "bob", 0.8)
            >>> extended.substitution.get("Y")
            'bob'
            >>> extended.score
            0.72  # 0.9 * 0.8
        """
        new_sub = self.substitution.bind(var, value)
        new_score = self.score * score  # Multiply scores (approximate joint probability)
        return CandidateBinding(substitution=new_sub, score=new_score)


def execute_atom_with_bindings(
    atom: Atom,
    binding: CandidateBinding,
    query_fn: Any,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Execute an atom query with partial variable bindings.

    Args:
        atom: Atom to query (may have variables)
        binding: Current partial binding of variables
        query_fn: Function to execute queries (signature: Query, k → results)
        k: Number of results to retrieve

    Returns:
        List of (value, score) tuples for the unbound variable

    Raises:
        ValueError: If atom has more than one unbound variable

    Example:
        >>> # Atom: parent(Y, Z) with binding {X: alice, Y: bob}
        >>> # Executes: parent(bob, Z)? → [(carol, 0.8), ...]
    """
    # Apply current bindings to atom
    partial_atom = binding.substitution.apply_to_atom(atom)

    # Check how many variables are still unbound
    unbound_vars = partial_atom.get_variables()

    if len(unbound_vars) == 0:
        # Atom is fully ground - can't query it (would need membership check)
        return []

    if len(unbound_vars) > 1:
        # Multiple unbound variables - not supported in Stage 4
        raise ValueError(
            f"Atom {partial_atom} has {len(unbound_vars)} unbound variables, "
            f"only 1 supported in Stage 4"
        )

    # Convert to Query
    query_args: list[str | None] = []
    for arg in partial_atom.args:
        if arg in unbound_vars:
            query_args.append(None)  # Variable
        else:
            query_args.append(arg)  # Ground constant

    query = Query(predicate=partial_atom.predicate, args=query_args)

    # Execute query
    result = query_fn(query, k)
    return result.results


def join_with_atom(
    candidates: list[CandidateBinding],
    atom: Atom,
    query_fn: Any,
    beam_width: int = 50,
    k: int = 10,
) -> list[CandidateBinding]:
    """Join current candidate bindings with a new atom using beam search.

    For each candidate binding, execute the atom query with partial bindings,
    extend the binding with results, and keep top beam_width candidates.

    Args:
        candidates: Current candidate bindings
        atom: Next atom to join
        query_fn: Function to execute queries
        beam_width: Maximum number of candidates to keep (beam search)
        k: Number of results per query

    Returns:
        Extended candidate bindings, sorted by score, limited to beam_width

    Example:
        >>> # Current: [{X: alice, score: 0.9}]
        >>> # Atom: parent(X, Y)
        >>> # After join: [{X: alice, Y: bob, score: 0.72}, ...]
    """
    # Get the unbound variable in the atom (after applying current bindings)
    # We'll execute queries to find values for this variable

    new_candidates: list[CandidateBinding] = []

    for candidate in candidates:
        # Execute atom query with current bindings
        try:
            results = execute_atom_with_bindings(atom, candidate, query_fn, k=k)
        except ValueError:
            # Atom has multiple unbound variables - skip for now
            continue

        if not results:
            # No results for this binding - dead end
            continue

        # Find which variable is unbound
        partial_atom = candidate.substitution.apply_to_atom(atom)
        unbound_vars = partial_atom.get_variables()

        if len(unbound_vars) != 1:
            continue

        unbound_var = unbound_vars[0]

        # Extend binding with each result
        for value, score in results:
            extended = candidate.extend(unbound_var, value, score)
            new_candidates.append(extended)

    # Beam search: keep only top beam_width candidates by score
    new_candidates.sort(key=lambda c: c.score, reverse=True)
    return new_candidates[:beam_width]


def initial_candidates_from_atom(
    atom: Atom,
    query_fn: Any,
    k: int = 10,
    kb: Any = None,
) -> list[CandidateBinding]:
    """Create initial candidate bindings from the first atom.

    Args:
        atom: First atom in rule body
        query_fn: Function to execute queries
        k: Number of results to retrieve
        kb: Knowledge base (needed for multi-variable atoms)

    Returns:
        List of initial candidate bindings

    Example:
        >>> # Atom: parent(X, Y)
        >>> # Returns: [{X: alice, Y: bob, score: 1.0}, ...]
    """
    # Get variables in atom
    variables = get_atom_unique_variables(atom)

    if len(variables) == 0:
        # Ground atom - create single binding with score 1.0
        return [CandidateBinding(substitution=Substitution(), score=1.0)]

    if len(variables) == 1:
        # Single variable - use query execution
        var = variables[0]

        # Convert atom to query
        query_args: list[str | None] = []
        for arg in atom.args:
            if arg == var:
                query_args.append(None)
            else:
                query_args.append(arg)

        query = Query(predicate=atom.predicate, args=query_args)
        result = query_fn(query, k)

        # Create candidate bindings from results
        candidates = []
        for value, score in result.results:
            sub = Substitution().bind(var, value)
            candidates.append(CandidateBinding(substitution=sub, score=score))

        return candidates

    # Multiple variables - enumerate all facts
    if kb is None:
        raise ValueError(
            f"First atom {atom} has {len(variables)} variables, "
            f"requires kb parameter for enumeration"
        )

    # Get all facts for this predicate
    facts = kb.get_facts(atom.predicate)

    if not facts:
        return []

    # Create bindings from facts
    candidates = []
    for fact in facts[:k]:  # Limit to k facts
        sub = Substitution()
        # Bind variables to fact values
        for i, arg in enumerate(atom.args):
            if arg in variables:
                if i < len(fact):
                    sub = sub.bind(arg, fact[i])

        candidates.append(CandidateBinding(substitution=sub, score=1.0))

    return candidates
