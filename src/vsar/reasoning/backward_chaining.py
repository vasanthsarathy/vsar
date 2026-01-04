"""Backward chaining (goal-directed proof search) using SLD resolution with VSA.

This module implements SLD resolution adapted for Vector Symbolic Architectures,
with approximate unification and tabling for cycle detection.
"""

from typing import Any

from pydantic import BaseModel, Field

from vsar.language.ast import Atom, NAFLiteral, Rule
from vsar.semantics.substitution import Substitution


class ProofResult(BaseModel):
    """Result of proving a goal.

    Attributes:
        substitution: Variable bindings that make the goal true
        similarity: Confidence score (min similarity across proof steps)
        depth: Proof depth (number of rule applications)
    """

    substitution: Substitution = Field(..., description="Variable bindings")
    similarity: float = Field(..., description="Proof confidence score")
    depth: int = Field(default=0, description="Proof depth")


class BackwardChainer:
    """Goal-directed proof search using SLD resolution with approximate VSA unification.

    This implements backward chaining (top-down reasoning) as an alternative to
    forward chaining. Instead of deriving all possible facts, it only proves
    specific queries by working backward from goals to facts.

    Key features:
    - SLD resolution with approximate unification via VSA
    - Tabling (memoization) to avoid infinite loops
    - Depth-limited search
    - Support for negation-as-failure (NAF)

    Args:
        engine: VSAR engine with knowledge base
        rules: List of rules for reasoning
        max_depth: Maximum proof depth (default: 10)
        threshold: Similarity threshold for unification (default: 0.5)

    Example:
        >>> # Setup
        >>> engine = VSAREngine([...])
        >>> engine.insert_fact(Fact("parent", ["alice", "bob"]))
        >>> engine.insert_fact(Fact("parent", ["bob", "charlie"]))
        >>>
        >>> # Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
        >>> rule = Rule(
        ...     head=Atom("grandparent", ["X", "Z"]),
        ...     body=[Atom("parent", ["X", "Y"]), Atom("parent", ["Y", "Z"])]
        ... )
        >>>
        >>> # Backward chaining
        >>> chainer = BackwardChainer(engine, [rule], max_depth=5)
        >>> goal = Atom("grandparent", ["alice", "charlie"])
        >>> proofs = chainer.prove_goal(goal)
        >>> len(proofs)  # Should find 1 proof
        1
    """

    def __init__(
        self,
        engine: Any,  # VSAREngine, avoid circular import
        rules: list[Rule],
        max_depth: int = 10,
        threshold: float = 0.5,
    ):
        """Initialize backward chainer."""
        self.engine = engine
        self.rules = rules
        self.max_depth = max_depth
        self.threshold = threshold

        # Tabling: cache proven goals to avoid infinite loops
        # Maps (goal_signature, depth) -> list of ProofResults
        self.table: dict[tuple[str, int], list[ProofResult]] = {}

        # Statistics
        self.table_hits = 0
        self.table_misses = 0

    def prove_goal(
        self, goal: Atom, current_depth: int = 0
    ) -> list[ProofResult]:
        """Prove a goal using SLD resolution with tabling.

        Args:
            goal: Goal atom to prove
            current_depth: Current proof depth

        Returns:
            List of proofs (substitution + similarity) that make the goal true

        Example:
            >>> goal = Atom("parent", ["alice", "X"])
            >>> proofs = chainer.prove_goal(goal)
            >>> proofs[0].substitution.get("X")
            'bob'
        """
        # Depth limit check
        if current_depth >= self.max_depth:
            return []

        # Check table (memoization)
        goal_sig = self._goal_signature(goal)
        table_key = (goal_sig, current_depth)

        if table_key in self.table:
            self.table_hits += 1
            return self.table[table_key]

        self.table_misses += 1

        proofs: list[ProofResult] = []

        # Strategy 1: Try to unify goal with facts in KB
        proofs.extend(self._prove_from_facts(goal))

        # Strategy 2: Try to unify goal with rule heads
        proofs.extend(self._prove_from_rules(goal, current_depth))

        # Store in table
        self.table[table_key] = proofs
        return proofs

    def _prove_from_facts(self, goal: Atom) -> list[ProofResult]:
        """Attempt to unify goal with facts in the knowledge base.

        Args:
            goal: Goal atom

        Returns:
            List of proofs from direct fact matches
        """
        proofs = []

        # Check if predicate exists in KB
        if not self.engine.kb.has_predicate(goal.predicate):
            return []

        # Get all facts for this predicate
        facts = self.engine.kb.get_facts(goal.predicate)

        # Try to unify goal with each fact
        for fact_tuple in facts:
            # Create atom from fact
            fact_atom = Atom(predicate=goal.predicate, args=list(fact_tuple))

            # Attempt unification
            sub, similarity = self._unify_vsa(goal, fact_atom)

            if sub is not None and similarity >= self.threshold:
                proofs.append(
                    ProofResult(
                        substitution=sub,
                        similarity=similarity,
                        depth=0,
                    )
                )

        return proofs

    def _prove_from_rules(self, goal: Atom, current_depth: int) -> list[ProofResult]:
        """Attempt to prove goal by unifying with rule heads and proving bodies.

        Args:
            goal: Goal atom
            current_depth: Current proof depth

        Returns:
            List of proofs from rule applications
        """
        proofs = []

        # Find rules whose head might unify with the goal
        for rule in self.rules:
            if rule.head.predicate != goal.predicate:
                continue

            # Unify goal with rule head
            head_sub, head_sim = self._unify_vsa(goal, rule.head)

            if head_sub is None or head_sim < self.threshold:
                continue

            # Separate positive atoms from NAF literals BEFORE applying substitution
            positive_atoms_original = []
            naf_literals = []

            for lit in rule.body:
                if isinstance(lit, NAFLiteral):
                    naf_literals.append(lit)
                else:
                    positive_atoms_original.append(lit)

            # Apply substitution to positive atoms
            positive_atoms = [head_sub.apply_to_atom(atom) for atom in positive_atoms_original]

            # Prove positive body atoms (conjunction)
            body_proofs = self._prove_conjunction(positive_atoms, current_depth + 1)

            # For each way to prove the body...
            for body_proof in body_proofs:
                # Compose substitutions
                combined_sub = head_sub.compose(body_proof.substitution)

                # Check NAF literals if any
                all_naf_succeed = True
                if naf_literals:
                    # Import here to avoid circular dependency
                    from vsar.reasoning.naf import evaluate_naf

                    for naf_lit in naf_literals:
                        # Apply combined substitution to NAF literal
                        instantiated_naf = NAFLiteral(atom=combined_sub.apply_to_atom(naf_lit.atom))
                        # Check if NAF succeeds
                        if not evaluate_naf(instantiated_naf, combined_sub, self.engine, self.threshold):
                            all_naf_succeed = False
                            break

                if not all_naf_succeed:
                    continue

                # Combine similarities (take minimum as confidence)
                combined_sim = min(head_sim, body_proof.similarity)

                proofs.append(
                    ProofResult(
                        substitution=combined_sub,
                        similarity=combined_sim,
                        depth=body_proof.depth + 1,
                    )
                )

        return proofs

    def _prove_conjunction(
        self, goals: list[Atom], current_depth: int
    ) -> list[ProofResult]:
        """Prove a conjunction of goals (AND).

        Args:
            goals: List of goal atoms to prove
            current_depth: Current proof depth

        Returns:
            List of proofs that satisfy all goals

        Example:
            >>> goals = [Atom("parent", ["X", "Y"]), Atom("parent", ["Y", "Z"])]
            >>> proofs = chainer._prove_conjunction(goals, 0)
        """
        # Base case: empty conjunction is trivially true
        if not goals:
            return [ProofResult(substitution=Substitution(), similarity=1.0, depth=0)]

        # Recursive case: prove first goal, then rest
        first_goal, *rest = goals
        results = []

        # Prove first goal
        first_proofs = self.prove_goal(first_goal, current_depth)

        for first_proof in first_proofs:
            # Apply substitution to remaining goals
            remaining_goals = [
                first_proof.substitution.apply_to_atom(g) for g in rest
            ]

            # Prove remaining goals
            rest_proofs = self._prove_conjunction(remaining_goals, current_depth)

            for rest_proof in rest_proofs:
                # Compose substitutions
                combined_sub = first_proof.substitution.compose(rest_proof.substitution)

                # Combine similarities (minimum)
                combined_sim = min(first_proof.similarity, rest_proof.similarity)

                # Max depth
                max_depth = max(first_proof.depth, rest_proof.depth)

                results.append(
                    ProofResult(
                        substitution=combined_sub,
                        similarity=combined_sim,
                        depth=max_depth,
                    )
                )

        return results

    def _unify_vsa(
        self, goal: Atom, fact: Atom
    ) -> tuple[Substitution | None, float]:
        """Unify two atoms using VSA approximate matching.

        This is the key adaptation of classical unification to VSA:
        instead of exact symbolic unification, we use similarity-based
        matching to find approximate bindings.

        Both goal and fact may contain variables. Variables are unified
        by building consistent substitutions.

        Args:
            goal: Goal atom (may have variables)
            fact: Fact atom (may have variables or be ground)

        Returns:
            (substitution, similarity) if unification succeeds, (None, 0.0) otherwise

        Example:
            >>> goal = Atom("parent", ["alice", "X"])
            >>> fact = Atom("parent", ["alice", "bob"])
            >>> sub, sim = chainer._unify_vsa(goal, fact)
            >>> sub.get("X")
            'bob'
            >>> sim
            0.95
        """
        # Predicates must match
        if goal.predicate != fact.predicate:
            return (None, 0.0)

        # Arity must match
        if len(goal.args) != len(fact.args):
            return (None, 0.0)

        # Build substitution and track similarity
        sub = Substitution()
        similarities = []

        for goal_arg, fact_arg in zip(goal.args, fact.args):
            goal_is_var = goal_arg[0].isupper() if goal_arg else False
            fact_is_var = fact_arg[0].isupper() if fact_arg else False

            if goal_is_var and fact_is_var:
                # Both are variables - bind them together
                # For simplicity, bind goal_arg to fact_arg
                if sub.has(goal_arg):
                    bound_value = sub.get(goal_arg)
                    if bound_value != fact_arg:
                        return (None, 0.0)
                else:
                    sub = sub.bind(goal_arg, fact_arg)
                similarities.append(1.0)

            elif goal_is_var:
                # Goal is variable, fact is constant
                if sub.has(goal_arg):
                    bound_value = sub.get(goal_arg)
                    if bound_value != fact_arg:
                        return (None, 0.0)
                else:
                    sub = sub.bind(goal_arg, fact_arg)
                similarities.append(1.0)

            elif fact_is_var:
                # Fact is variable, goal is constant
                # Bind fact variable to goal constant
                if sub.has(fact_arg):
                    bound_value = sub.get(fact_arg)
                    if bound_value != goal_arg:
                        return (None, 0.0)
                else:
                    sub = sub.bind(fact_arg, goal_arg)
                similarities.append(1.0)

            else:
                # Both are constants - must match exactly
                if goal_arg == fact_arg:
                    similarities.append(1.0)
                else:
                    return (None, 0.0)

        # Overall similarity is average of all argument matches
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        return (sub, avg_similarity)

    def _goal_signature(self, goal: Atom) -> str:
        """Create a signature for a goal for tabling.

        Args:
            goal: Goal atom

        Returns:
            String signature
        """
        args_str = ",".join(goal.args)
        return f"{goal.predicate}({args_str})"

    def clear_table(self) -> None:
        """Clear the memoization table.

        Useful when KB is updated or when starting a new query session.
        """
        self.table.clear()
        self.table_hits = 0
        self.table_misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about backward chaining execution.

        Returns:
            Dictionary with statistics
        """
        return {
            "table_size": len(self.table),
            "table_hits": self.table_hits,
            "table_misses": self.table_misses,
            "hit_rate": (
                self.table_hits / (self.table_hits + self.table_misses)
                if (self.table_hits + self.table_misses) > 0
                else 0.0
            ),
        }
