"""Knowledge base persistence (save/load)."""

import json
from pathlib import Path

import h5py
import jax.numpy as jnp

from vsar.kb.store import KnowledgeBase
from vsar.kernel.base import KernelBackend


def save_kb(kb: KnowledgeBase, path: Path | str) -> None:
    """
    Save knowledge base to HDF5 file.

    Saves both predicate bundles (hypervectors) and fact lists.

    File structure:
    - /bundles/<predicate> - bundled hypervector datasets
    - /facts/<predicate> - JSON-encoded fact lists (attributes)

    Args:
        kb: Knowledge base to save
        path: Path to HDF5 file

    Example:
        >>> save_kb(kb, "knowledge_base.h5")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Create groups
        bundles_group = f.create_group("bundles")
        facts_group = f.create_group("facts")

        # Save each predicate's bundle and facts
        for predicate in kb.predicates():
            # Save bundle as dataset
            bundle = kb.get_bundle(predicate)
            if bundle is not None:
                bundles_group.create_dataset(predicate, data=bundle)

            # Save facts as JSON attribute
            facts = kb.get_facts(predicate)
            facts_json = json.dumps(facts)
            facts_group.attrs[predicate] = facts_json


def load_kb(backend: KernelBackend, path: Path | str) -> KnowledgeBase:
    """
    Load knowledge base from HDF5 file.

    Args:
        backend: Kernel backend for the loaded KB
        path: Path to HDF5 file

    Returns:
        Loaded knowledge base

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        >>> kb = load_kb(backend, "knowledge_base.h5")
        >>> kb.count()
        10
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"KB file not found: {path}")

    kb = KnowledgeBase(backend)

    with h5py.File(path, "r") as f:
        bundles_group = f["bundles"]
        facts_group = f["facts"]

        # Load each predicate
        for predicate in bundles_group.keys():
            # Load bundle
            bundle = jnp.array(bundles_group[predicate][:])

            # Load facts
            facts_json = facts_group.attrs[predicate]
            facts = json.loads(facts_json)

            # Reconstruct KB state
            kb._bundles[predicate] = bundle
            kb._facts[predicate] = [tuple(fact) for fact in facts]

    return kb
