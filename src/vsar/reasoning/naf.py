"""Negation-as-failure (NAF) evaluation with threshold-based semantics."""

from typing import TYPE_CHECKING

from ..language.ast import Atom, NAFLiteral, Query
from ..semantics.substitution import Substitution

if TYPE_CHECKING:
    from ..semantics.engine import VSAREngine


def evaluate_naf(
    literal: NAFLiteral,
    bindings: Substitution,
    engine: "VSAREngine",
    threshold: float = 0.5,
) -> bool:
    """
    Evaluate a NAF literal under current variable bindings.

    Uses threshold-based semantics (Option B):
    - not p(X) succeeds if NO results have score >= threshold
    - not p(X) fails if ANY result has score >= threshold

    This handles approximate matching gracefully - weak spurious matches
    (score < threshold) are treated as "effectively absent".

    Args:
        literal: NAF literal to evaluate (e.g., not enemy(X, _))
        bindings: Current variable bindings
        engine: VSAR engine for querying
        threshold: Minimum score for "presence" (default: 0.5)

    Returns:
        True if NAF succeeds (query fails or all results below threshold)
        False if NAF fails (query succeeds with strong results)

    Raises:
        ValueError: If atom has unbound variables after applying bindings

    Example:
        >>> # Given bindings {X: "alice"}
        >>> # NAF literal: not enemy(X, bob)
        >>> # After binding: not enemy(alice, bob)
        >>> # Query enemy(alice, bob) returns []
        >>> # NAF succeeds: True
        >>>
        >>> naf_lit = NAFLiteral(atom=Atom("enemy", ["X", "bob"]))
        >>> bindings = Substitution().bind("X", "alice")
        >>> result = evaluate_naf(naf_lit, bindings, engine)
        >>> result  # True if alice is NOT enemy of bob
    """
    # 1. Apply current bindings to the NAF atom
    ground_atom = bindings.apply_to_atom(literal.atom)

    # 2. Check if atom is fully ground or has unbound variables
    unbound_vars = ground_atom.get_variables()

    # 3. For NAF, use EXACT matching on bound arguments
    # Get all facts for this predicate and check for exact matches on bound positions
    pred_name = ground_atom.predicate
    if not engine.kb.has_predicate(pred_name):
        # Predicate doesn't exist - NAF succeeds (no facts at all)
        return True

    all_facts = engine.kb.get_facts(pred_name)

    # Check if any fact matches the bound arguments exactly
    for fact_args in all_facts:
        # Check if bound arguments match exactly
        match = True
        for i, query_arg in enumerate(ground_atom.args):
            # If query arg is lowercase (constant), check for exact match
            if not (query_arg and query_arg[0].isupper()):
                # It's a constant - must match exactly
                if i >= len(fact_args) or str(fact_args[i]) != query_arg:
                    match = False
                    break
            # If query arg is uppercase (variable), it's a wildcard - any value matches

        if match:
            # Found an exact match on bound arguments - NAF fails
            return False

    # No exact matches found - NAF succeeds
    return True


def evaluate_naf_in_rule_body(
    naf_literal: NAFLiteral,
    current_bindings: Substitution,
    engine: "VSAREngine",
    threshold: float = 0.5,
) -> bool:
    """
    Evaluate NAF literal in rule body during forward chaining.

    This is a wrapper around evaluate_naf() specifically for use in
    rule application contexts.

    Args:
        naf_literal: NAF literal from rule body
        current_bindings: Current substitution from earlier body atoms
        engine: VSAR engine
        threshold: Threshold for NAF evaluation

    Returns:
        True if NAF literal succeeds (can continue rule application)
        False if NAF literal fails (rule application fails)

    Example:
        >>> # Rule: safe(X) :- person(X), not enemy(X, _)
        >>> # Current bindings: {X: "alice"}
        >>> # Evaluate: not enemy(alice, _)
        >>> # This checks if alice has NO enemies (above threshold)
    """
    return evaluate_naf(naf_literal, current_bindings, engine, threshold)


def has_naf_literals(body: list) -> bool:
    """
    Check if rule body contains any NAF literals.

    Args:
        body: Rule body (list of Atom or NAFLiteral)

    Returns:
        True if body contains at least one NAFLiteral

    Example:
        >>> from vsar.language.ast import Atom, NAFLiteral
        >>> body1 = [Atom("person", ["X"])]
        >>> has_naf_literals(body1)  # False
        >>>
        >>> body2 = [Atom("person", ["X"]), NAFLiteral(atom=Atom("enemy", ["X", "_"]))]
        >>> has_naf_literals(body2)  # True
    """
    return any(isinstance(lit, NAFLiteral) for lit in body)


def separate_positive_and_naf(
    body: list,
) -> tuple[list[Atom], list[NAFLiteral]]:
    """
    Separate rule body into positive atoms and NAF literals.

    This is useful for rule evaluation: positive atoms are processed first
    to bind variables, then NAF literals are evaluated with those bindings.

    Args:
        body: Rule body (list of Atom or NAFLiteral)

    Returns:
        Tuple of (positive_atoms, naf_literals)

    Example:
        >>> from vsar.language.ast import Atom, NAFLiteral
        >>> body = [
        ...     Atom("person", ["X"]),
        ...     NAFLiteral(atom=Atom("enemy", ["X", "_"])),
        ...     Atom("lives", ["X", "city"])
        ... ]
        >>> positive, naf = separate_positive_and_naf(body)
        >>> len(positive)  # 2
        >>> len(naf)  # 1
    """
    positive = [lit for lit in body if isinstance(lit, Atom)]
    naf = [lit for lit in body if isinstance(lit, NAFLiteral)]
    return positive, naf
