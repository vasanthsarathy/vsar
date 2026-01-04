"""Experiment 2: Scalability Test

Tests VSAR performance as the knowledge base scales from 100 to 1M facts.

Metrics:
- Query time (ms) - should be O(d), constant w.r.t. n
- Retrieval accuracy (precision@1, precision@5)
- Memory usage

Expected results:
- Query time remains constant (vectorized operations don't scale with KB size)
- Accuracy remains stable up to 100K facts
- Possible degradation at 1M facts due to increased entity overlap

Scales tested: 100, 1K, 10K, 100K, 1M facts
"""

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.synthetic_data_generator import generate_scalability_dataset
from vsar.encoding.role_filler_encoder import RoleFillerEncoder
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.retrieval.query import Retriever
from vsar.symbols.registry import SymbolRegistry


@dataclass
class ScalabilityResult:
    """Results for a single scale."""
    n_facts: int
    n_queries: int
    query_time_ms_mean: float
    query_time_ms_std: float
    precision_at_1: float
    precision_at_5: float
    total_encoding_time_ms: float
    avg_similarity: float


def run_scalability_test(
    n_facts: int,
    predicate: str = "related",
    arity: int = 2,
    dim: int = 8192,
    seed: int = 42,
) -> ScalabilityResult:
    """Run scalability test at a specific scale.

    Args:
        n_facts: Number of facts to test
        predicate: Predicate name
        arity: Predicate arity
        dim: Hypervector dimension
        seed: Random seed

    Returns:
        ScalabilityResult with performance metrics
    """
    print(f"\n{'='*80}")
    print(f"Testing scale: {n_facts:,} facts")
    print(f"{'='*80}")

    # Generate synthetic dataset
    print("  Generating dataset...", end=" ", flush=True)
    start = time.time()
    facts, test_queries = generate_scalability_dataset(
        n_facts=n_facts,
        predicate=predicate,
        arity=arity,
        seed=seed,
    )
    gen_time = (time.time() - start) * 1000
    print(f"Done ({gen_time:.0f}ms)")

    # Initialize VSAR components
    print("  Initializing VSAR...", end=" ", flush=True)
    backend = FHRRBackend(dim=dim, seed=seed)
    registry = SymbolRegistry(dim=dim, seed=seed)
    encoder = RoleFillerEncoder(backend, registry, seed=seed)
    kb = KnowledgeBase(backend)
    retriever = Retriever(backend, registry, kb, encoder)
    print("Done")

    # Encode and insert all facts
    print(f"  Encoding {len(facts):,} facts...", end=" ", flush=True)
    start = time.time()
    for fact in facts:
        atom_vec = encoder.encode_atom(predicate, list(fact))
        kb.insert(predicate, atom_vec, fact)
    encoding_time = (time.time() - start) * 1000
    print(f"Done ({encoding_time:.0f}ms, {encoding_time/len(facts):.2f}ms/fact)")

    # Run test queries
    print(f"  Running {len(test_queries)} test queries...", end=" ", flush=True)
    query_times = []
    precisions_at_1 = []
    precisions_at_5 = []
    similarities = []

    for ground_truth_fact, query_position, bound_args in test_queries:
        # Time the query
        start = time.time()
        results = retriever.retrieve(
            predicate=predicate,
            var_position=query_position,
            bound_args=bound_args,
            k=5,
        )
        query_time = (time.time() - start) * 1000
        query_times.append(query_time)

        # Evaluate precision
        expected_entity = ground_truth_fact[query_position - 1]

        # Precision@1
        if results and results[0][0] == expected_entity:
            precisions_at_1.append(1.0)
            similarities.append(results[0][1])
        else:
            precisions_at_1.append(0.0)
            if results:
                similarities.append(results[0][1])

        # Precision@5
        retrieved_entities = [entity for entity, _ in results[:5]]
        if expected_entity in retrieved_entities:
            precisions_at_5.append(1.0)
        else:
            precisions_at_5.append(0.0)

    print("Done")

    # Compute statistics
    result = ScalabilityResult(
        n_facts=n_facts,
        n_queries=len(test_queries),
        query_time_ms_mean=float(np.mean(query_times)),
        query_time_ms_std=float(np.std(query_times)),
        precision_at_1=float(np.mean(precisions_at_1)),
        precision_at_5=float(np.mean(precisions_at_5)),
        total_encoding_time_ms=encoding_time,
        avg_similarity=float(np.mean(similarities)) if similarities else 0.0,
    )

    # Print summary
    print(f"\n  Results:")
    print(f"    Query time:     {result.query_time_ms_mean:.2f} ± {result.query_time_ms_std:.2f} ms")
    print(f"    Precision@1:    {result.precision_at_1:.3f}")
    print(f"    Precision@5:    {result.precision_at_5:.3f}")
    print(f"    Avg similarity: {result.avg_similarity:.3f}")

    return result


def print_results_table(results: list[ScalabilityResult]):
    """Print results in formatted table."""
    print("\n" + "=" * 100)
    print("SCALABILITY TEST RESULTS")
    print("=" * 100)
    print()
    print(f"{'Scale':<12} {'Query Time (ms)':<20} {'P@1':<10} {'P@5':<10} {'Avg Sim':<10}")
    print("-" * 100)

    for r in results:
        scale_str = f"{r.n_facts:,}"
        query_time_str = f"{r.query_time_ms_mean:.2f} ± {r.query_time_ms_std:.2f}"
        print(f"{scale_str:<12} {query_time_str:<20} {r.precision_at_1:<10.3f} "
              f"{r.precision_at_5:<10.3f} {r.avg_similarity:<10.3f}")

    print("=" * 100)


def save_results_csv(results: list[ScalabilityResult], output_path: Path):
    """Save results to CSV for paper figures."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'n_facts', 'n_queries', 'query_time_ms_mean', 'query_time_ms_std',
            'precision_at_1', 'precision_at_5', 'total_encoding_time_ms', 'avg_similarity'
        ])

        for r in results:
            writer.writerow([
                r.n_facts, r.n_queries, r.query_time_ms_mean, r.query_time_ms_std,
                r.precision_at_1, r.precision_at_5, r.total_encoding_time_ms, r.avg_similarity
            ])

    print(f"\nResults saved to: {output_path}")


def main():
    """Run full scalability experiment."""
    print("\n" + "=" * 100)
    print("EXPERIMENT 2: SCALABILITY TEST")
    print("=" * 100)

    # Scales to test
    scales = [100, 1_000, 10_000, 100_000]

    # Note: 1M facts takes ~15 minutes to encode and requires ~16GB RAM
    # Uncomment to test at 1M scale:
    # scales.append(1_000_000)

    # Run tests
    results = []
    for n_facts in scales:
        result = run_scalability_test(
            n_facts=n_facts,
            predicate="related",
            arity=2,
            dim=8192,
            seed=42,
        )
        results.append(result)

    # Print summary
    print_results_table(results)

    # Check if query time is constant (coefficient of variation < 0.3)
    query_times = [r.query_time_ms_mean for r in results]
    cv = np.std(query_times) / np.mean(query_times)
    print(f"\nQuery time coefficient of variation: {cv:.3f}")
    if cv < 0.3:
        print("✓ Query time is approximately constant w.r.t. KB size (as expected)")
    else:
        print("⚠ Query time varies with KB size (unexpected)")

    # Check if accuracy is stable
    p1_values = [r.precision_at_1 for r in results]
    p1_drop = p1_values[0] - p1_values[-1]
    print(f"\nPrecision@1 drop: {p1_drop:.3f} ({p1_values[0]:.3f} → {p1_values[-1]:.3f})")
    if p1_drop < 0.1:
        print("✓ Accuracy remains stable across scales")
    else:
        print("⚠ Accuracy degrades at large scales")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    save_results_csv(results, results_dir / "scalability.csv")


if __name__ == "__main__":
    main()
