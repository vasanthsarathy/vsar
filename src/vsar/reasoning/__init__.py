"""Reasoning engines for VSAR - query answering, forward chaining, and consistency checking."""

from .query_engine import QueryEngine, QueryResult
from .consistency import ConsistencyChecker, ConsistencyReport, Contradiction
from .stratification import check_stratification, StratificationResult, Dependency, DependencyGraph

__all__ = [
    "QueryEngine",
    "QueryResult",
    "ConsistencyChecker",
    "ConsistencyReport",
    "Contradiction",
    "check_stratification",
    "StratificationResult",
    "Dependency",
    "DependencyGraph",
]
