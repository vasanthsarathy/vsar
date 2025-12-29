"""Basis vector generation and persistence for VSAR symbols."""

from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp

from vsar.kernel.base import KernelBackend

from .spaces import SymbolSpace


def generate_basis(
    space: SymbolSpace, name: str, backend: KernelBackend, seed: int
) -> jnp.ndarray:
    """
    Generate a deterministic basis vector for a symbol.

    The basis vector is generated using the backend's random generation with
    a deterministic seed derived from the space, name, and base seed. This
    ensures that the same symbol always gets the same hypervector.

    Args:
        space: Symbol space (E, R, A, etc.)
        name: Symbol name (e.g., "alice", "parent")
        backend: Kernel backend for vector generation
        seed: Base random seed for reproducibility

    Returns:
        Deterministic basis hypervector

    Example:
        >>> from vsar.kernel import FHRRBackend
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> vec1 = generate_basis(SymbolSpace.ENTITIES, "alice", backend, 42)
        >>> vec2 = generate_basis(SymbolSpace.ENTITIES, "alice", backend, 42)
        >>> assert jnp.allclose(vec1, vec2)  # Deterministic
    """
    # Create deterministic seed from space, name, and base seed
    # Use hash to convert string to integer
    space_hash = hash(space.value)
    name_hash = hash(name)
    combined_seed = (seed + space_hash + name_hash) % (2**31 - 1)

    # Generate basis vector with deterministic key
    key = jax.random.PRNGKey(combined_seed)
    vec = backend.generate_random(key, (backend.dimension,))

    # Normalize to ensure unit vectors
    return backend.normalize(vec)


def save_basis(path: Path, basis: dict[tuple[SymbolSpace, str], jnp.ndarray]) -> None:
    """
    Save basis vectors to HDF5 file.

    The basis is saved with metadata including the symbol space, name, and
    vector data. This enables reproducibility across sessions.

    Args:
        path: Path to HDF5 file
        basis: Dictionary mapping (space, name) to hypervector

    Example:
        >>> basis = {
        ...     (SymbolSpace.ENTITIES, "alice"): vec1,
        ...     (SymbolSpace.ENTITIES, "bob"): vec2,
        ... }
        >>> save_basis(Path("basis.h5"), basis)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Save metadata
        f.attrs["version"] = "0.1.0"
        f.attrs["num_symbols"] = len(basis)

        # Save each symbol's basis vector
        for (space, name), vec in basis.items():
            # Create group for symbol
            group_name = f"{space.value}/{name}"
            group = f.create_group(group_name)

            # Store vector data
            group.create_dataset("vector", data=vec)
            group.attrs["space"] = space.value
            group.attrs["name"] = name


def load_basis(path: Path) -> dict[tuple[SymbolSpace, str], jnp.ndarray]:
    """
    Load basis vectors from HDF5 file.

    Args:
        path: Path to HDF5 file

    Returns:
        Dictionary mapping (space, name) to hypervector

    Raises:
        FileNotFoundError: If basis file doesn't exist

    Example:
        >>> basis = load_basis(Path("basis.h5"))
        >>> vec = basis[(SymbolSpace.ENTITIES, "alice")]
    """
    if not path.exists():
        raise FileNotFoundError(f"Basis file not found: {path}")

    basis: dict[tuple[SymbolSpace, str], jnp.ndarray] = {}

    with h5py.File(path, "r") as f:
        # Recursively visit all groups
        def visit_group(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group) and "vector" in obj:
                # This is a symbol group
                space_str = obj.attrs["space"]
                symbol_name = obj.attrs["name"]

                # Convert space string back to enum
                space = SymbolSpace(space_str)

                # Load vector as JAX array
                vec = jnp.array(obj["vector"][:])

                basis[(space, symbol_name)] = vec

        f.visititems(visit_group)

    return basis
