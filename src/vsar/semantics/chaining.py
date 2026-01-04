"""Forward chaining for rule-based derivation."""

import warnings
from typing import Any

from pydantic import BaseModel, Field

from vsar.language.ast import Atom, NAFLiteral, Rule
from vsar.reasoning.stratification import check_stratification
from vsar.semantics.engine import VSAREngine


class ChainingResult(BaseModel):
    """Result of forward chaining execution.

    Attributes:
        iterations: Number of iterations performed
        total_derived: Total number of facts derived
        fixpoint_reached: True if fixpoint reached (no new facts)
        max_iterations_reached: True if stopped due to max iterations
        derived_per_iteration: List of facts derived in each iteration
    """

    iterations: int = Field(..., description="Number of iterations performed")
    total_derived: int = Field(..., description="Total facts derived")
    fixpoint_reached: bool = Field(..., description="Fixpoint reached")
    max_iterations_reached: bool = Field(..., description="Max iterations reached")
    derived_per_iteration: list[int] = Field(
        ..., description="Facts derived per iteration"
    )


def apply_rules(
    engine: VSAREngine,
    rules: list[Rule],
    max_iterations: int = 100,
    k: int | None = None,
    semi_naive: bool = True,
) -> ChainingResult:
    """Apply rules iteratively until fixpoint or max iterations.

    Supports both naive and semi-naive evaluation strategies:
    - Naive: Re-applies all rules to all facts every iteration
    - Semi-naive: Only applies rules when body predicates have new facts

    Semi-naive evaluation is significantly faster for transitive closure
    and recursive rules, as it avoids redundant re-evaluation.

    Args:
        engine: VSAR engine with knowledge base
        rules: List of rules to apply
        max_iterations: Maximum number of iterations (default: 100)
        k: Number of results per query (default: 10)
        semi_naive: Use semi-naive optimization (default: True)

    Returns:
        ChainingResult with statistics

    Example:
        >>> # Base facts: parent(alice, bob), parent(bob, carol)
        >>> # Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
        >>> result = apply_rules(engine, [grandparent_rule])
        >>> result.total_derived  # 1 (alice, carol)
        >>> result.fixpoint_reached  # True
    """
    if k is None:
        k = 10

    # Check stratification and warn if non-stratified
    stratification = check_stratification(rules)
    if not stratification.is_stratified:
        warning_msg = (
            "Warning: Non-stratified program detected. "
            "Negation-as-failure may have unpredictable semantics.\n"
            f"{stratification.summary()}"
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)

    iteration = 0
    total_derived = 0
    derived_per_iteration: list[int] = []
    iteration_derived = 0  # Track last iteration's derived count

    # Semi-naive: track predicates with new facts
    new_predicates: set[str] = set()
    if semi_naive:
        # Initially, all predicates have "new" facts (the base facts)
        new_predicates = set(engine.kb.predicates())

    while iteration < max_iterations:
        iteration_derived = 0
        current_iteration_new_predicates: set[str] = set()

        # Apply each rule
        for rule in rules:
            # Semi-naive optimization: skip rule if no body predicates have new facts
            if semi_naive:
                # Extract predicates from body (handle both Atoms and NAFLiterals)
                rule_body_predicates = set()
                for lit in rule.body:
                    if isinstance(lit, Atom):
                        rule_body_predicates.add(lit.predicate)
                    elif isinstance(lit, NAFLiteral):
                        rule_body_predicates.add(lit.atom.predicate)

                if not rule_body_predicates.intersection(new_predicates):
                    # No new facts in any body predicate - skip this rule
                    continue

            # Track KB state before applying rule
            predicate_counts_before = {
                pred: engine.kb.count(pred) for pred in engine.kb.predicates()
            }

            # Apply rule
            count = engine.apply_rule(rule, k=k)
            iteration_derived += count

            # Track which predicates got new facts (for next iteration)
            if semi_naive and count > 0:
                # Check which predicates changed
                for pred in engine.kb.predicates():
                    count_after = engine.kb.count(pred)
                    count_before = predicate_counts_before.get(pred, 0)
                    if count_after > count_before:
                        current_iteration_new_predicates.add(pred)

        # Update new predicates for next iteration
        if semi_naive:
            new_predicates = current_iteration_new_predicates

        # Check for fixpoint FIRST (before recording iteration)
        if iteration_derived == 0:
            # No new facts derived - fixpoint reached
            break

        # Only record productive iterations (where facts were derived)
        iteration += 1
        derived_per_iteration.append(iteration_derived)
        total_derived += iteration_derived

    # Determine why we stopped
    fixpoint_reached = iteration_derived == 0
    max_iterations_reached = iteration >= max_iterations and not fixpoint_reached

    return ChainingResult(
        iterations=iteration,
        total_derived=total_derived,
        fixpoint_reached=fixpoint_reached,
        max_iterations_reached=max_iterations_reached,
        derived_per_iteration=derived_per_iteration,
    )
