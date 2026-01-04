"""Stratification analysis for safe negation-as-failure.

Stratification ensures that negation is used safely in logic programs by
organizing rules into layers (strata) where predicates only depend negatively
on predicates in lower strata.

This prevents logical paradoxes from circular negative dependencies like:
    rule p(X) :- not q(X).
    rule q(X) :- not p(X).
"""

from dataclasses import dataclass
from typing import Set

from vsar.language.ast import Atom, NAFLiteral, Rule


@dataclass
class Dependency:
    """A dependency between two predicates."""

    source: str  # Predicate that depends on target
    target: str  # Predicate being depended upon
    negative: bool  # True if this is a negative dependency (NAF)

    def __repr__(self) -> str:
        arrow = "-/→" if self.negative else "→"
        return f"{self.source} {arrow} {self.target}"


@dataclass
class StratificationResult:
    """Result of stratification analysis."""

    is_stratified: bool
    strata: dict[str, int]  # Predicate -> stratum number
    cycles: list[list[str]]  # List of cycles involving negative dependencies
    dependencies: list[Dependency]  # All dependencies

    def __repr__(self) -> str:
        if self.is_stratified:
            return f"✓ Stratified ({len(self.strata)} predicates, {max(self.strata.values()) + 1 if self.strata else 0} strata)"
        else:
            return f"✗ Non-stratified ({len(self.cycles)} negative cycles)"

    def summary(self) -> str:
        """Human-readable summary of stratification."""
        if self.is_stratified:
            # Group predicates by stratum
            by_stratum: dict[int, list[str]] = {}
            for pred, stratum in self.strata.items():
                if stratum not in by_stratum:
                    by_stratum[stratum] = []
                by_stratum[stratum].append(pred)

            lines = ["✓ Program is stratified:"]
            for stratum in sorted(by_stratum.keys()):
                preds = ", ".join(sorted(by_stratum[stratum]))
                lines.append(f"  Stratum {stratum}: {preds}")
            return "\n".join(lines)
        else:
            lines = ["✗ Program is NOT stratified - contains negative cycles:"]
            for i, cycle in enumerate(self.cycles, 1):
                cycle_str = " → ".join(cycle + [cycle[0]])
                lines.append(f"  Cycle {i}: {cycle_str}")
            lines.append("\nWarning: Non-stratified programs may have unpredictable semantics.")
            return "\n".join(lines)


class DependencyGraph:
    """Dependency graph for stratification analysis."""

    def __init__(self):
        """Initialize empty dependency graph."""
        self.dependencies: list[Dependency] = []
        self.predicates: Set[str] = set()

    def add_rule(self, rule: Rule) -> None:
        """Add dependencies from a rule.

        Args:
            rule: Rule to analyze
        """
        head_pred = rule.head.predicate
        self.predicates.add(head_pred)

        # Extract dependencies from rule body
        for body_lit in rule.body:
            if isinstance(body_lit, Atom):
                # Positive dependency
                target_pred = body_lit.predicate
                self.predicates.add(target_pred)
                self.dependencies.append(
                    Dependency(source=head_pred, target=target_pred, negative=False)
                )
            elif isinstance(body_lit, NAFLiteral):
                # Negative dependency (NAF)
                target_pred = body_lit.atom.predicate
                self.predicates.add(target_pred)
                self.dependencies.append(
                    Dependency(source=head_pred, target=target_pred, negative=True)
                )

    def find_negative_cycles(self) -> list[list[str]]:
        """Find cycles that involve at least one negative dependency.

        Uses DFS to detect cycles and checks if they contain negative edges.

        Returns:
            List of cycles (each cycle is a list of predicates)
        """
        cycles = []

        # Build adjacency lists
        positive_edges: dict[str, list[str]] = {p: [] for p in self.predicates}
        negative_edges: dict[str, list[str]] = {p: [] for p in self.predicates}

        for dep in self.dependencies:
            if dep.negative:
                negative_edges[dep.source].append(dep.target)
            else:
                positive_edges[dep.source].append(dep.target)

        # Find strongly connected components that contain negative edges
        visited = set()
        rec_stack = set()
        current_path = []

        def dfs(node: str) -> None:
            """DFS to find cycles."""
            visited.add(node)
            rec_stack.add(node)
            current_path.append(node)

            # Check both positive and negative edges
            all_neighbors = positive_edges[node] + negative_edges[node]
            for neighbor in all_neighbors:
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle - check if it has negative edges
                    cycle_start_idx = current_path.index(neighbor)
                    cycle = current_path[cycle_start_idx:]

                    # Check if this cycle contains a negative edge
                    has_negative = False
                    for i in range(len(cycle)):
                        src = cycle[i]
                        tgt = cycle[(i + 1) % len(cycle)]
                        if tgt in negative_edges[src]:
                            has_negative = True
                            break

                    if has_negative and cycle not in cycles:
                        cycles.append(cycle[:])

            current_path.pop()
            rec_stack.remove(node)

        for pred in self.predicates:
            if pred not in visited:
                dfs(pred)

        return cycles

    def compute_strata(self) -> dict[str, int]:
        """Compute stratum for each predicate.

        Uses a fixed-point algorithm:
        - Start with all predicates at stratum 0
        - For each negative dependency P → Q, ensure stratum(P) > stratum(Q)
        - Iterate until fixpoint

        Returns:
            Dictionary mapping predicate -> stratum number
            Returns empty dict if program is non-stratified
        """
        # Check for negative cycles first
        if self.find_negative_cycles():
            return {}

        # Initialize all predicates to stratum 0
        strata = {pred: 0 for pred in self.predicates}

        # Fixed-point iteration
        changed = True
        max_iterations = len(self.predicates) * 2
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for dep in self.dependencies:
                if dep.negative:
                    # Negative dependency: source must be in higher stratum than target
                    if strata[dep.source] <= strata[dep.target]:
                        strata[dep.source] = strata[dep.target] + 1
                        changed = True
                else:
                    # Positive dependency: source must be >= target stratum
                    if strata[dep.source] < strata[dep.target]:
                        strata[dep.source] = strata[dep.target]
                        changed = True

        return strata


def check_stratification(rules: list[Rule]) -> StratificationResult:
    """Check if a set of rules is stratified.

    Args:
        rules: List of rules to check

    Returns:
        StratificationResult with stratification info and any negative cycles

    Example:
        >>> # Stratified program
        >>> rules = [
        ...     Rule(head=Atom("safe", ["X"]),
        ...          body=[Atom("person", ["X"]), NAFLiteral(Atom("enemy", ["X", "_"]))])
        ... ]
        >>> result = check_stratification(rules)
        >>> result.is_stratified  # True

        >>> # Non-stratified program
        >>> rules = [
        ...     Rule(head=Atom("p", ["X"]), body=[NAFLiteral(Atom("q", ["X"]))]),
        ...     Rule(head=Atom("q", ["X"]), body=[NAFLiteral(Atom("p", ["X"]))])
        ... ]
        >>> result = check_stratification(rules)
        >>> result.is_stratified  # False
        >>> len(result.cycles)  # 1
    """
    # Build dependency graph
    graph = DependencyGraph()
    for rule in rules:
        graph.add_rule(rule)

    # Find negative cycles
    cycles = graph.find_negative_cycles()

    # Compute stratification
    strata = graph.compute_strata()

    is_stratified = len(cycles) == 0

    return StratificationResult(
        is_stratified=is_stratified,
        strata=strata,
        cycles=cycles,
        dependencies=graph.dependencies,
    )
