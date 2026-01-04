"""Synthetic data generator for scalability experiments.

Generates large knowledge bases with controlled properties:
- Configurable number of facts
- Binary, ternary, or quaternary predicates
- Unique entity generation
- Ground truth for query validation
"""

import random
from typing import Literal


class SyntheticKBGenerator:
    """Generate synthetic knowledge bases for scalability testing."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.rng = random.Random(seed)
        self._entity_counter = 0

    def _generate_entity(self, prefix: str = "e") -> str:
        """Generate a unique entity name."""
        entity = f"{prefix}{self._entity_counter}"
        self._entity_counter += 1
        return entity

    def generate_binary_facts(
        self,
        predicate: str,
        n_facts: int,
        reuse_entities: bool = True,
    ) -> list[tuple[str, str]]:
        """Generate binary predicate facts.

        Args:
            predicate: Predicate name (e.g., "parent")
            n_facts: Number of facts to generate
            reuse_entities: If True, reuse entities to create dense KB
                           If False, each fact uses unique entities

        Returns:
            List of (arg1, arg2) tuples

        Example:
            >>> gen = SyntheticKBGenerator(seed=42)
            >>> facts = gen.generate_binary_facts("parent", 100, reuse_entities=True)
            >>> len(facts)
            100
        """
        facts = []

        if reuse_entities:
            # Create a pool of entities to reuse (sqrt(n_facts) is a good heuristic)
            pool_size = max(10, int(n_facts ** 0.5))
            entity_pool = [self._generate_entity() for _ in range(pool_size)]

            for _ in range(n_facts):
                arg1 = self.rng.choice(entity_pool)
                arg2 = self.rng.choice(entity_pool)
                # Avoid self-loops
                while arg2 == arg1:
                    arg2 = self.rng.choice(entity_pool)
                facts.append((arg1, arg2))
        else:
            # Each fact uses unique entities
            for _ in range(n_facts):
                arg1 = self._generate_entity()
                arg2 = self._generate_entity()
                facts.append((arg1, arg2))

        return facts

    def generate_ternary_facts(
        self,
        predicate: str,
        n_facts: int,
        reuse_entities: bool = True,
    ) -> list[tuple[str, str, str]]:
        """Generate ternary predicate facts.

        Args:
            predicate: Predicate name (e.g., "teaches")
            n_facts: Number of facts to generate
            reuse_entities: If True, reuse entities to create dense KB

        Returns:
            List of (arg1, arg2, arg3) tuples
        """
        facts = []

        if reuse_entities:
            pool_size = max(10, int(n_facts ** (1/3)))
            entity_pool = [self._generate_entity() for _ in range(pool_size)]

            for _ in range(n_facts):
                arg1 = self.rng.choice(entity_pool)
                arg2 = self.rng.choice(entity_pool)
                arg3 = self.rng.choice(entity_pool)
                facts.append((arg1, arg2, arg3))
        else:
            for _ in range(n_facts):
                arg1 = self._generate_entity()
                arg2 = self._generate_entity()
                arg3 = self._generate_entity()
                facts.append((arg1, arg2, arg3))

        return facts

    def generate_quaternary_facts(
        self,
        predicate: str,
        n_facts: int,
        reuse_entities: bool = True,
    ) -> list[tuple[str, str, str, str]]:
        """Generate quaternary predicate facts.

        Args:
            predicate: Predicate name (e.g., "transaction")
            n_facts: Number of facts to generate
            reuse_entities: If True, reuse entities to create dense KB

        Returns:
            List of (arg1, arg2, arg3, arg4) tuples
        """
        facts = []

        if reuse_entities:
            pool_size = max(10, int(n_facts ** 0.25))
            entity_pool = [self._generate_entity() for _ in range(pool_size)]

            for _ in range(n_facts):
                arg1 = self.rng.choice(entity_pool)
                arg2 = self.rng.choice(entity_pool)
                arg3 = self.rng.choice(entity_pool)
                arg4 = self.rng.choice(entity_pool)
                facts.append((arg1, arg2, arg3, arg4))
        else:
            for _ in range(n_facts):
                arg1 = self._generate_entity()
                arg2 = self._generate_entity()
                arg3 = self._generate_entity()
                arg4 = self._generate_entity()
                facts.append((arg1, arg2, arg3, arg4))

        return facts

    def generate_facts(
        self,
        predicate: str,
        arity: int,
        n_facts: int,
        reuse_entities: bool = True,
    ) -> list[tuple[str, ...]]:
        """Generate facts of any arity.

        Args:
            predicate: Predicate name
            arity: Predicate arity (2, 3, or 4)
            n_facts: Number of facts to generate
            reuse_entities: If True, reuse entities

        Returns:
            List of fact tuples
        """
        if arity == 2:
            return self.generate_binary_facts(predicate, n_facts, reuse_entities)
        elif arity == 3:
            return self.generate_ternary_facts(predicate, n_facts, reuse_entities)
        elif arity == 4:
            return self.generate_quaternary_facts(predicate, n_facts, reuse_entities)
        else:
            raise ValueError(f"Unsupported arity: {arity}")

    def generate_test_queries(
        self,
        facts: list[tuple[str, ...]],
        n_queries: int,
    ) -> list[tuple[tuple[str, ...], int, dict[str, str]]]:
        """Generate test queries from a fact set.

        Args:
            facts: List of fact tuples
            n_queries: Number of queries to generate

        Returns:
            List of (fact, query_position, bound_args) tuples
            where:
            - fact: The ground truth fact
            - query_position: Position being queried (1-indexed)
            - bound_args: Dict mapping position (as string) to entity name
        """
        queries = []
        arity = len(facts[0])

        for _ in range(n_queries):
            # Pick a random fact
            fact = self.rng.choice(facts)

            # Pick a random position to query
            query_position = self.rng.randint(1, arity)

            # Create bound args (all positions except query_position)
            bound_args = {
                str(i+1): fact[i]
                for i in range(arity)
                if (i+1) != query_position
            }

            queries.append((fact, query_position, bound_args))

        return queries


def generate_scalability_dataset(
    n_facts: int,
    predicate: str = "related",
    arity: int = 2,
    seed: int = 42,
) -> tuple[list[tuple[str, ...]], list[tuple[tuple[str, ...], int, dict[str, str]]]]:
    """Generate a complete scalability test dataset.

    Args:
        n_facts: Number of facts to generate
        predicate: Predicate name
        arity: Predicate arity
        seed: Random seed

    Returns:
        (facts, test_queries) tuple
    """
    generator = SyntheticKBGenerator(seed=seed)
    facts = generator.generate_facts(predicate, arity, n_facts, reuse_entities=True)
    queries = generator.generate_test_queries(facts, n_queries=min(100, n_facts))

    return facts, queries


if __name__ == "__main__":
    # Test the generator
    print("Testing Synthetic KB Generator\n")

    gen = SyntheticKBGenerator(seed=42)

    # Test binary facts
    print("Binary facts (n=10):")
    binary_facts = gen.generate_binary_facts("parent", 10, reuse_entities=True)
    for fact in binary_facts[:5]:
        print(f"  parent{fact}")

    # Test ternary facts
    print("\nTernary facts (n=10):")
    gen._entity_counter = 0  # Reset
    ternary_facts = gen.generate_ternary_facts("teaches", 10, reuse_entities=True)
    for fact in ternary_facts[:5]:
        print(f"  teaches{fact}")

    # Test query generation
    print("\nTest queries (n=5):")
    queries = gen.generate_test_queries(binary_facts, 5)
    for fact, pos, bound in queries:
        print(f"  Fact: {fact}, Query pos: {pos}, Bound: {bound}")

    # Test scalability dataset
    print("\nScalability dataset (n=100):")
    facts, queries = generate_scalability_dataset(100, "related", arity=2, seed=42)
    print(f"  Generated {len(facts)} facts")
    print(f"  Generated {len(queries)} test queries")
