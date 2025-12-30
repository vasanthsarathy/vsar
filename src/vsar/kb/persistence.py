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

    Saves both fact vectors (hypervectors) and fact tuples separately.

    File structure:
    - /vectors/<predicate>/<index> - individual fact vector datasets
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
        vectors_group = f.create_group("vectors")
        facts_group = f.create_group("facts")

        # Save each predicate's vectors and facts
        for predicate in kb.predicates():
            # Save vectors as separate datasets in predicate subgroup
            vectors = kb.get_vectors(predicate)
            if vectors:
                pred_group = vectors_group.create_group(predicate)
                for i, vec in enumerate(vectors):
                    pred_group.create_dataset(str(i), data=vec)

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
        vectors_group = f["vectors"]
        facts_group = f["facts"]

        # Load each predicate
        for predicate in vectors_group.keys():
            # Load vectors from predicate subgroup
            pred_group = vectors_group[predicate]
            vectors = []
            for i in range(len(pred_group.keys())):
                vec = jnp.array(pred_group[str(i)][:])
                vectors.append(vec)

            # Load facts
            facts_json = facts_group.attrs[predicate]
            facts = json.loads(facts_json)

            # Reconstruct KB state
            kb._vectors[predicate] = vectors
            kb._facts[predicate] = [tuple(fact) for fact in facts]

    return kb
