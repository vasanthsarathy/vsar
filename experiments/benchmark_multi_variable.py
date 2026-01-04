"""Benchmark multi-variable query accuracy vs single-variable baseline.

This script measures the accuracy of multi-variable queries (e.g., parent(?, ?))
compared to the ground truth established by single-variable queries.

Results are saved to experiments/results/multi_variable_accuracy.csv
"""

import csv
from pathlib import Path

from vsar.language.ast import Directive, Fact, Query
from vsar.semantics.engine import VSAREngine


def create_engine(seed=42):
    """Create a VSAR engine with standard configuration."""
    directives = [
        Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": seed}),
        Directive(name="threshold", params={"value": 0.22}),
        Directive(name="beam", params={"width": 50}),
        Directive(name="novelty", params={"threshold": 0.95}),
    ]
    return VSAREngine(directives)


def benchmark_family_tree():
    """Benchmark multi-variable queries on family tree dataset."""
    print("=" * 70)
    print("BENCHMARK: Family Tree (Binary Predicates)")
    print("=" * 70)

    engine = create_engine(seed=42)

    # Insert parent facts
    parent_facts = [
        ("alice", "bob"),
        ("alice", "charlie"),
        ("bob", "david"),
        ("bob", "eve"),
        ("charlie", "frank"),
        ("charlie", "grace"),
        ("david", "henry"),
        ("eve", "iris"),
    ]

    for parent, child in parent_facts:
        engine.insert_fact(Fact(predicate="parent", args=[parent, child]))

    print(f"\nInserted {len(parent_facts)} parent facts")
    print(f"Knowledge base: {engine.stats()}")

    # Ground truth: Use single-variable queries
    print("\n" + "-" * 70)
    print("GROUND TRUTH (Single-Variable Queries)")
    print("-" * 70)

    known_parents = set(p for p, _ in parent_facts)
    ground_truth_pairs = set()

    for parent in known_parents:
        result = engine.query(Query(predicate="parent", args=[parent, None]), k=10)
        for child, score in result.results:
            ground_truth_pairs.add((parent, child))
            print(f"  {parent} -> {child} (similarity: {score:.3f})")

    print(f"\nTotal ground truth pairs: {len(ground_truth_pairs)}")

    # Multi-variable query
    print("\n" + "-" * 70)
    print("MULTI-VARIABLE QUERY: parent(?, ?)")
    print("-" * 70)

    multi_result = engine.query(Query(predicate="parent", args=[None, None]), k=20)

    print(f"Retrieved {len(multi_result.results)} results:\n")

    multi_pairs = set()
    import ast

    for res_str, score in multi_result.results:
        pair = ast.literal_eval(res_str)
        multi_pairs.add(pair)
        in_gt = pair in ground_truth_pairs
        status = "[OK]" if in_gt else "[FAIL]"
        print(f"  {status} {pair[0]} -> {pair[1]} (similarity: {score:.3f})")

    # Calculate accuracy metrics
    print("\n" + "-" * 70)
    print("ACCURACY METRICS")
    print("-" * 70)

    true_positives = len(multi_pairs & ground_truth_pairs)
    false_positives = len(multi_pairs - ground_truth_pairs)
    false_negatives = len(ground_truth_pairs - multi_pairs)

    precision = true_positives / len(multi_pairs) if multi_pairs else 0
    recall = true_positives / len(ground_truth_pairs) if ground_truth_pairs else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"True Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"\nPrecision: {precision:.1%}")
    print(f"Recall:    {recall:.1%}")
    print(f"F1 Score:  {f1:.1%}")

    return {
        "dataset": "family_tree",
        "arity": 2,
        "num_facts": len(parent_facts),
        "ground_truth_size": len(ground_truth_pairs),
        "retrieved": len(multi_pairs),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def benchmark_works_in():
    """Benchmark multi-variable queries on ternary relation dataset."""
    print("\n\n" + "=" * 70)
    print("BENCHMARK: Works-In (Ternary Predicates)")
    print("=" * 70)

    engine = create_engine(seed=42)

    # Insert works_in facts (person, department, role)
    works_in_facts = [
        ("alice", "engineering", "lead"),
        ("bob", "engineering", "engineer"),
        ("charlie", "engineering", "intern"),
        ("david", "sales", "manager"),
        ("eve", "sales", "rep"),
        ("frank", "marketing", "director"),
    ]

    for person, dept, role in works_in_facts:
        engine.insert_fact(Fact(predicate="works_in", args=[person, dept, role]))

    print(f"\nInserted {len(works_in_facts)} works_in facts")

    # Test multi-variable query with one bound argument
    print("\n" + "-" * 70)
    print("MULTI-VARIABLE QUERY: works_in(?, engineering, ?)")
    print("-" * 70)

    result = engine.query(
        Query(predicate="works_in", args=[None, "engineering", None]), k=10
    )

    print(f"Retrieved {len(result.results)} results:\n")

    import ast

    retrieved_pairs = set()
    for res_str, score in result.results:
        pair = ast.literal_eval(res_str)  # (person, role)
        retrieved_pairs.add(pair)
        print(f"  {pair[0]} -> {pair[1]} (similarity: {score:.3f})")

    # Ground truth: people in engineering with their roles
    ground_truth_engineering = {
        ("alice", "lead"),
        ("bob", "engineer"),
        ("charlie", "intern"),
    }

    accuracy = (
        len(retrieved_pairs & ground_truth_engineering) / len(ground_truth_engineering)
        if ground_truth_engineering
        else 0
    )

    print(f"\nAccuracy: {accuracy:.1%}")

    # Convert accuracy to precision/recall/f1 format
    tp = len(retrieved_pairs & ground_truth_engineering)
    fp = len(retrieved_pairs - ground_truth_engineering)
    fn = len(ground_truth_engineering - retrieved_pairs)

    precision = tp / len(retrieved_pairs) if retrieved_pairs else 0
    recall = tp / len(ground_truth_engineering) if ground_truth_engineering else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "dataset": "works_in",
        "arity": 3,
        "num_facts": len(works_in_facts),
        "ground_truth_size": len(ground_truth_engineering),
        "retrieved": len(retrieved_pairs),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    """Run all benchmarks and save results."""
    results = []

    # Run benchmarks
    results.append(benchmark_family_tree())
    results.append(benchmark_works_in())

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / "multi_variable_accuracy.csv"

    with open(output_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
