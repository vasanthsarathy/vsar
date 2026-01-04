"""Experiment 1: Encoding Method Comparison

Compares four encoding strategies for VSA-based reasoning:
1. Role-filler: ARG1 ⊗ t1 ⊕ ARG2 ⊗ t2 (traditional VSA)
2. Shift-based: shift(t1,1) + shift(t2,2) (positional encoding, no predicate)
3. Hybrid (no cancel): P ⊗ (shift(t1,1) + shift(t2,2)) (predicate + shift)
4. Hybrid + cancel: Same as #3 with interference cancellation during retrieval

Tests on predicates with different arities:
- Binary: parent(person, child)
- Ternary: teaches(professor, course, semester)
- Quaternary: transaction(buyer, seller, item, price)

Metrics:
- Average similarity score for correct retrievals
- Standard deviation
- 95% confidence intervals

Expected results (from KEY_INSIGHTS.md):
- Role-filler: 0.26 (binary), 0.20 (ternary), 0.15 (quaternary)
- Shift-based: 0.63 (binary), 0.50 (ternary), 0.40 (quaternary)
- Hybrid (no cancel): 0.70 (binary), 0.55 (ternary), 0.45 (quaternary)
- Hybrid + cancel: 0.93 (binary), 0.82 (ternary), 0.75 (quaternary)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np

from vsar.kernel.vsa_backend import FHRRBackend
from vsar.kb.store import KnowledgeBase
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


@dataclass
class EncodingResult:
    """Results for a single encoding method on a dataset."""
    encoding_method: str
    predicate: str
    arity: int
    num_facts: int
    num_queries: int
    similarities: list[float]
    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float


class PureRoleFillerEncoder:
    """Pure role-filler encoding without predicate binding.

    Formula: enc(p(t1,...,tk)) = ARG1 ⊗ t1 ⊕ ARG2 ⊗ t2 ⊕ ... ⊕ ARGk ⊗ tk

    Does NOT include predicate in the encoding - relies on KB partitioning.
    """

    def __init__(self, backend, registry):
        self.backend = backend
        self.registry = registry

    def encode_atom(self, predicate: str, args: list[str]) -> jnp.ndarray:
        """Encode atom using pure role-filler binding."""
        bound_args = []
        for i, arg in enumerate(args):
            position = i + 1
            # Get role vector
            role_vec = self.registry.register(SymbolSpace.ARG_ROLES, f"ARG{position}")
            # Get entity vector
            entity_vec = self.registry.register(SymbolSpace.ENTITIES, arg)
            # Bind role with entity
            bound = self.backend.bind(role_vec, entity_vec)
            bound_args.append(bound)

        # Bundle all bound arguments (use plain sum)
        atom_vec = sum(bound_args)
        return self.backend.normalize(atom_vec)

    def decode_position(self, atom_vec: jnp.ndarray, position: int,
                       bound_args: dict[str, str] = None) -> jnp.ndarray:
        """Decode a position from the atom vector."""
        # Get role vector for this position
        role_vec = self.registry.register(SymbolSpace.ARG_ROLES, f"ARG{position}")
        # Unbind the role to get entity
        entity_vec = self.backend.unbind(atom_vec, role_vec)
        return entity_vec


class PureShiftEncoder:
    """Pure shift-based encoding without predicate binding.

    Formula: enc(p(t1,...,tk)) = shift(t1,1) + shift(t2,2) + ... + shift(tk,k)

    Does NOT include predicate - different predicates with same args have same encoding.
    """

    def __init__(self, backend, registry):
        self.backend = backend
        self.registry = registry

    def encode_atom(self, predicate: str, args: list[str]) -> jnp.ndarray:
        """Encode atom using pure shift-based encoding."""
        shifted_args = []
        for i, arg in enumerate(args):
            position = i + 1
            # Get entity vector
            entity_vec = self.registry.register(SymbolSpace.ENTITIES, arg)
            # Shift by position
            shifted = self.backend.permute(entity_vec, position)
            shifted_args.append(shifted)

        # Bundle all shifted arguments (use plain sum for linear superposition)
        atom_vec = sum(shifted_args)
        return self.backend.normalize(atom_vec)

    def decode_position(self, atom_vec: jnp.ndarray, position: int,
                       bound_args: dict[str, str] = None) -> jnp.ndarray:
        """Decode a position from the atom vector."""
        # Shift decode
        entity_vec = self.backend.permute(atom_vec, -position)
        return entity_vec


class HybridEncoder:
    """Hybrid encoding with predicate binding and shift-based positions.

    Formula: enc(p(t1,...,tk)) = P_p ⊗ (shift(t1,1) + shift(t2,2) + ... + shift(tk,k))

    Combines:
    - Predicate distinguishability (from P_p binding)
    - Clean positional encoding (from shifts)
    """

    def __init__(self, backend, registry):
        self.backend = backend
        self.registry = registry

    def encode_atom(self, predicate: str, args: list[str]) -> jnp.ndarray:
        """Encode atom using hybrid encoding."""
        # Get predicate vector
        pred_vec = self.registry.register(SymbolSpace.PREDICATES, predicate)

        # Encode arguments with shifts
        shifted_args = []
        for i, arg in enumerate(args):
            position = i + 1
            entity_vec = self.registry.register(SymbolSpace.ENTITIES, arg)
            shifted = self.backend.permute(entity_vec, position)
            shifted_args.append(shifted)

        # Bundle shifted arguments
        args_bundle = sum(shifted_args)

        # Bind predicate with args bundle
        atom_vec = self.backend.bind(pred_vec, args_bundle)
        return self.backend.normalize(atom_vec)

    def decode_position(self, atom_vec: jnp.ndarray, predicate: str, position: int,
                       bound_args: dict[str, str] = None,
                       use_cancellation: bool = False) -> jnp.ndarray:
        """Decode a position with optional interference cancellation."""
        # Unbind predicate
        pred_vec = self.registry.register(SymbolSpace.PREDICATES, predicate)
        args_bundle = self.backend.unbind(atom_vec, pred_vec)

        # Apply interference cancellation if requested
        if use_cancellation and bound_args:
            for pos_str, entity_name in bound_args.items():
                pos = int(pos_str)
                if pos != position:  # Don't cancel the position we're decoding
                    entity_vec = self.registry.register(SymbolSpace.ENTITIES, entity_name)
                    shifted_contribution = self.backend.permute(entity_vec, pos)
                    args_bundle = args_bundle - shifted_contribution

        # Shift decode
        entity_vec = self.backend.permute(args_bundle, -position)
        return entity_vec


def load_dataset(dataset_path: Path) -> tuple[str, list[tuple[str, ...]]]:
    """Load dataset from JSONL file.

    Returns:
        (predicate_name, list of fact tuples)
    """
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]

    if not data:
        raise ValueError(f"Empty dataset: {dataset_path}")

    # Extract predicate and facts
    predicate = data[0]['predicate']
    facts = [tuple(item['args']) for item in data]

    return predicate, facts


def run_encoding_test(
    encoding_method: Literal['role_filler', 'shift', 'hybrid', 'hybrid_cancel'],
    predicate: str,
    facts: list[tuple[str, ...]],
    dim: int = 8192,
    seed: int = 42,
    num_queries: int = 100,
) -> EncodingResult:
    """Run encoding test on a dataset.

    Args:
        encoding_method: Which encoding to use
        predicate: Predicate name
        facts: List of fact tuples
        dim: Hypervector dimension
        seed: Random seed
        num_queries: Number of queries to test

    Returns:
        EncodingResult with statistics
    """
    # Initialize backend and registry
    backend = FHRRBackend(dim=dim, seed=seed)
    registry = SymbolRegistry(dim=dim, seed=seed)

    # Create encoder based on method
    if encoding_method == 'role_filler':
        encoder = PureRoleFillerEncoder(backend, registry)
    elif encoding_method == 'shift':
        encoder = PureShiftEncoder(backend, registry)
    else:  # hybrid or hybrid_cancel
        encoder = HybridEncoder(backend, registry)

    # Encode and store all facts
    fact_vectors = []
    for fact in facts:
        atom_vec = encoder.encode_atom(predicate, list(fact))
        fact_vectors.append((atom_vec, fact))

    # Generate test queries
    # For each fact, query with all but one argument bound
    arity = len(facts[0])
    similarities = []

    rng = np.random.RandomState(seed)

    for _ in range(min(num_queries, len(facts))):
        # Pick a random fact
        fact_idx = rng.randint(0, len(facts))
        fact_vec, fact = fact_vectors[fact_idx]

        # Pick a random position to query
        query_position = rng.randint(1, arity + 1)

        # Build bound args (all positions except query_position)
        bound_args = {
            str(i+1): fact[i]
            for i in range(arity)
            if (i+1) != query_position
        }

        # Decode the query position
        if encoding_method in ['role_filler', 'shift']:
            decoded_vec = encoder.decode_position(fact_vec, query_position, bound_args)
        else:  # hybrid
            use_cancel = (encoding_method == 'hybrid_cancel')
            decoded_vec = encoder.decode_position(
                fact_vec, predicate, query_position, bound_args, use_cancellation=use_cancel
            )

        # Cleanup to find the entity
        results = registry.cleanup(SymbolSpace.ENTITIES, decoded_vec, k=1)

        if results:
            retrieved_entity, similarity = results[0]
            expected_entity = fact[query_position - 1]

            # Check if we got the right entity
            if retrieved_entity == expected_entity:
                similarities.append(float(similarity))

    # Compute statistics
    if not similarities:
        return EncodingResult(
            encoding_method=encoding_method,
            predicate=predicate,
            arity=arity,
            num_facts=len(facts),
            num_queries=0,
            similarities=[],
            mean=0.0,
            std=0.0,
            ci_95_lower=0.0,
            ci_95_upper=0.0,
        )

    mean = float(np.mean(similarities))
    std = float(np.std(similarities))
    sem = std / np.sqrt(len(similarities))
    ci_margin = 1.96 * sem  # 95% CI

    return EncodingResult(
        encoding_method=encoding_method,
        predicate=predicate,
        arity=arity,
        num_facts=len(facts),
        num_queries=len(similarities),
        similarities=similarities,
        mean=mean,
        std=std,
        ci_95_lower=mean - ci_margin,
        ci_95_upper=mean + ci_margin,
    )


def print_results_table(results: list[EncodingResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("ENCODING COMPARISON RESULTS")
    print("=" * 100)
    print()

    # Group by arity
    by_arity = {}
    for r in results:
        if r.arity not in by_arity:
            by_arity[r.arity] = []
        by_arity[r.arity].append(r)

    for arity in sorted(by_arity.keys()):
        arity_name = {2: "Binary", 3: "Ternary", 4: "Quaternary"}.get(arity, f"{arity}-ary")
        print(f"\n{arity_name} Predicates:")
        print("-" * 100)
        print(f"{'Encoding Method':<20} {'Mean Score':<12} {'Std Dev':<12} {'95% CI':<25} {'Queries':<10}")
        print("-" * 100)

        for r in by_arity[arity]:
            ci_str = f"[{r.ci_95_lower:.4f}, {r.ci_95_upper:.4f}]"
            print(f"{r.encoding_method:<20} {r.mean:<12.4f} {r.std:<12.4f} {ci_str:<25} {r.num_queries:<10}")

    print("\n" + "=" * 100)


def save_results_csv(results: list[EncodingResult], output_path: Path):
    """Save results to CSV for paper figures."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'encoding_method', 'predicate', 'arity', 'num_facts', 'num_queries',
            'mean', 'std', 'ci_95_lower', 'ci_95_upper'
        ])

        for r in results:
            writer.writerow([
                r.encoding_method, r.predicate, r.arity, r.num_facts, r.num_queries,
                r.mean, r.std, r.ci_95_lower, r.ci_95_upper
            ])

    print(f"\nResults saved to: {output_path}")


def main():
    """Run the full encoding comparison experiment."""
    # Dataset paths
    datasets_dir = Path(__file__).parent / "datasets"
    datasets = [
        datasets_dir / "family_tree_100.jsonl",  # Binary predicate
        datasets_dir / "academic_200.jsonl",      # Ternary predicate
        datasets_dir / "transactions_200.jsonl",   # Quaternary predicate
    ]

    # Encoding methods to test
    methods = ['role_filler', 'shift', 'hybrid', 'hybrid_cancel']

    # Run experiments
    all_results = []

    for dataset_path in datasets:
        if not dataset_path.exists():
            print(f"WARNING: Dataset not found: {dataset_path}")
            print(f"         Skipping...")
            continue

        print(f"\nLoading dataset: {dataset_path.name}")
        predicate, facts = load_dataset(dataset_path)
        arity = len(facts[0])

        print(f"  Predicate: {predicate}")
        print(f"  Arity: {arity}")
        print(f"  Facts: {len(facts)}")

        for method in methods:
            print(f"\n  Testing {method}...")
            result = run_encoding_test(
                encoding_method=method,
                predicate=predicate,
                facts=facts,
                dim=8192,
                seed=42,
                num_queries=100,
            )
            all_results.append(result)
            print(f"    Mean similarity: {result.mean:.4f} ± {result.std:.4f}")

    # Print summary table
    print_results_table(all_results)

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    save_results_csv(all_results, results_dir / "encoding_comparison.csv")


if __name__ == "__main__":
    main()
