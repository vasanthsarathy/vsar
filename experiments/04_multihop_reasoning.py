"""Experiment 4: Multi-Hop Reasoning

Tests VSAR's ability to perform multi-hop transitive reasoning using rules.

Setup:
- Base facts: parent(X, Y) relationships (3-generation family tree)
- Rule: ancestor(X, Z) :- parent(X, Z)
- Rule: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)

Queries:
- 1-hop: Direct parents (ancestor should match parent)
- 2-hop: Grandparents
- 3-hop: Great-grandparents

Metrics:
- Precision: Fraction of retrieved results that are correct
- Recall: Fraction of ground truth results that were retrieved
- F1 score: Harmonic mean of precision and recall

Expected results (from plan):
- 1-hop: P≈0.95, R≈0.98, F1≈0.96
- 2-hop: P≈0.88, R≈0.92, F1≈0.90
- 3-hop: P≈0.80, R≈0.85, F1≈0.82
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from vsar.language.ast import Atom, Directive, Fact, Query, Rule
from vsar.semantics.chaining import apply_rules
from vsar.semantics.engine import VSAREngine


@dataclass
class HopResult:
    """Results for a specific hop depth."""
    hop: int
    n_queries: int
    precision: float
    recall: float
    f1_score: float
    avg_similarity: float
    n_ground_truth: int
    n_retrieved: int
    n_correct: int


def generate_family_tree(n_generations: int = 3) -> list[tuple[str, str]]:
    """Generate a multi-generation family tree.

    Args:
        n_generations: Number of generations (1 = just parents)

    Returns:
        List of (parent, child) tuples

    Structure:
        Gen 0: alice, bob
        Gen 1: carol, dave (children of alice, bob respectively)
        Gen 2: emma, frank (children of carol, dave respectively)
        etc.
    """
    facts = []

    # Generation 0 (roots)
    gen_0 = ["alice", "bob"]

    # Build subsequent generations
    current_gen = gen_0
    child_counter = ord('c')  # Start from 'c' (carol)

    for gen_num in range(1, n_generations + 1):
        next_gen = []

        for parent in current_gen:
            # Each person has 2 children
            child1 = chr(child_counter)
            child2 = chr(child_counter + 1)
            child_counter += 2

            facts.append((parent, child1))
            facts.append((parent, child2))

            next_gen.extend([child1, child2])

        current_gen = next_gen

    return facts


def compute_ground_truth_ancestors(
    parent_facts: list[tuple[str, str]]
) -> dict[tuple[str, str], int]:
    """Compute all ancestor relationships with hop distances.

    Args:
        parent_facts: List of (parent, child) tuples

    Returns:
        Dictionary mapping (ancestor, descendant) -> hop_distance
    """
    # Build parent mapping
    children = defaultdict(list)
    for parent, child in parent_facts:
        children[parent].append(child)

    # BFS to find all ancestors with distances
    ancestors = {}

    for parent, child in parent_facts:
        # Direct parent-child is 1-hop
        ancestors[(parent, child)] = 1

        # Find all descendants of this child with BFS
        queue = [(child, 1)]
        visited = {child}

        while queue:
            current, dist = queue.pop(0)

            # Parent is ancestor of all current's descendants at dist+1
            for grandchild in children[current]:
                if grandchild not in visited:
                    visited.add(grandchild)
                    ancestors[(parent, grandchild)] = dist + 1
                    queue.append((grandchild, dist + 1))

    return ancestors


def run_multihop_experiment(
    parent_facts: list[tuple[str, str]],
    max_hops: int = 3,
    dim: int = 8192,
    seed: int = 42,
    threshold: float = 0.22,
) -> list[HopResult]:
    """Run multi-hop reasoning experiment.

    Args:
        parent_facts: Base parent facts
        max_hops: Maximum hop depth to test
        dim: Hypervector dimension
        seed: Random seed
        threshold: Similarity threshold for reasoning

    Returns:
        List of HopResult for each hop depth
    """
    print(f"\n{'='*80}")
    print(f"MULTI-HOP REASONING EXPERIMENT")
    print(f"{'='*80}")
    print(f"  Base facts: {len(parent_facts)}")
    print(f"  Max hops: {max_hops}")
    print(f"  Dimension: {dim}")
    print(f"  Threshold: {threshold}")

    # Initialize VSAR engine
    directives = [
        Directive(name="model", params={"type": "FHRR", "dim": dim, "seed": seed}),
        Directive(name="novelty", params={"threshold": 0.95}),
        Directive(name="beam", params={"width": 50}),
    ]
    engine = VSAREngine(directives)

    # Insert parent facts
    print(f"\n  Inserting parent facts...", end=" ", flush=True)
    for parent, child in parent_facts:
        engine.insert_fact(Fact(predicate="parent", args=[parent, child]))
    print("Done")

    # Create rules for ancestor
    rules = [
        Rule(
            head=Atom(predicate="ancestor", args=["X", "Z"]),
            body=[Atom(predicate="parent", args=["X", "Z"])],
        ),
        Rule(
            head=Atom(predicate="ancestor", args=["X", "Z"]),
            body=[
                Atom(predicate="parent", args=["X", "Y"]),
                Atom(predicate="ancestor", args=["Y", "Z"]),
            ],
        ),
    ]

    # Run forward chaining
    print(f"  Running forward chaining...", end=" ", flush=True)
    result = apply_rules(engine, rules, max_iterations=max_hops + 1, k=50)
    print(f"Done (derived {result.total_derived} facts in {result.iterations} iterations)")

    # Compute ground truth
    print(f"  Computing ground truth...", end=" ", flush=True)
    ground_truth = compute_ground_truth_ancestors(parent_facts)
    print(f"Done ({len(ground_truth)} ancestor pairs)")

    # Test each hop depth
    results = []

    for hop in range(1, max_hops + 1):
        print(f"\n  Testing {hop}-hop queries...")

        # Get ground truth for this hop
        hop_ground_truth = {
            (anc, desc): dist
            for (anc, desc), dist in ground_truth.items()
            if dist == hop
        }

        if not hop_ground_truth:
            print(f"    No ground truth for {hop}-hop")
            continue

        # Query ancestor for each ground truth pair
        n_correct = 0
        n_retrieved = 0
        total_similarity = 0.0
        n_queries = 0

        for ancestor, descendant in hop_ground_truth.keys():
            # Query: ancestor(ancestor, X) should retrieve descendant
            query = Query(predicate="ancestor", args=[ancestor, None])
            query_result = engine.query(query, k=10)

            n_queries += 1

            # Check if descendant is in results
            if query_result.results:
                retrieved_entities = [entity for entity, _ in query_result.results]
                similarities = [score for _, score in query_result.results]

                if descendant in retrieved_entities:
                    n_correct += 1
                    # Get similarity for this result
                    idx = retrieved_entities.index(descendant)
                    total_similarity += similarities[idx]

                n_retrieved += len(retrieved_entities)

        # Compute metrics
        precision = n_correct / n_retrieved if n_retrieved > 0 else 0.0
        recall = n_correct / len(hop_ground_truth) if hop_ground_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_sim = total_similarity / n_correct if n_correct > 0 else 0.0

        result = HopResult(
            hop=hop,
            n_queries=n_queries,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_similarity=avg_sim,
            n_ground_truth=len(hop_ground_truth),
            n_retrieved=n_retrieved,
            n_correct=n_correct,
        )

        results.append(result)

        print(f"    Precision: {precision:.3f}")
        print(f"    Recall:    {recall:.3f}")
        print(f"    F1:        {f1:.3f}")
        print(f"    Avg sim:   {avg_sim:.3f}")
        print(f"    Correct:   {n_correct}/{len(hop_ground_truth)}")

    return results


def print_results_table(results: list[HopResult]):
    """Print results in formatted table."""
    print("\n" + "=" * 100)
    print("MULTI-HOP REASONING RESULTS")
    print("=" * 100)
    print()
    print(f"{'Hop':<6} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Avg Sim':<12} {'Correct/Total':<15}")
    print("-" * 100)

    for r in results:
        print(f"{r.hop:<6} {r.precision:<12.3f} {r.recall:<12.3f} {r.f1_score:<12.3f} "
              f"{r.avg_similarity:<12.3f} {r.n_correct}/{r.n_ground_truth:<10}")

    print("=" * 100)

    # Check degradation pattern
    if len(results) >= 2:
        f1_drop = results[0].f1_score - results[-1].f1_score
        print(f"\nF1 degradation: {results[0].f1_score:.3f} -> {results[-1].f1_score:.3f} "
              f"(drop: {f1_drop:.3f})")

        if f1_drop < 0.2:
            print("OK: Graceful degradation (F1 drop < 0.2)")
        else:
            print("WARNING: Significant degradation (F1 drop >= 0.2)")


def save_results_csv(results: list[HopResult], output_path: Path):
    """Save results to CSV for paper figures."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'hop', 'n_queries', 'precision', 'recall', 'f1_score',
            'avg_similarity', 'n_ground_truth', 'n_retrieved', 'n_correct'
        ])

        for r in results:
            writer.writerow([
                r.hop, r.n_queries, r.precision, r.recall, r.f1_score,
                r.avg_similarity, r.n_ground_truth, r.n_retrieved, r.n_correct
            ])

    print(f"\nResults saved to: {output_path}")


def main():
    """Run the multi-hop reasoning experiment."""
    # Generate 3-generation family tree
    parent_facts = generate_family_tree(n_generations=3)

    print(f"Generated family tree:")
    print(f"  Total parent facts: {len(parent_facts)}")
    print(f"  Sample facts: {parent_facts[:5]}")

    # Run experiment
    results = run_multihop_experiment(
        parent_facts=parent_facts,
        max_hops=3,
        dim=8192,
        seed=42,
        threshold=0.22,
    )

    # Print summary
    print_results_table(results)

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    save_results_csv(results, results_dir / "multihop_reasoning.csv")


if __name__ == "__main__":
    main()
