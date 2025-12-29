"""Unbinding operations for retrieval."""

import jax.numpy as jnp

from vsar.kernel.base import KernelBackend


def unbind_query_from_bundle(
    query_vec: jnp.ndarray,
    kb_bundle: jnp.ndarray,
    backend: KernelBackend,
) -> jnp.ndarray:
    """
    Unbind query pattern from KB bundle.

    This extracts the relevant facts from the bundle by unbinding
    the query pattern, leaving the variable bindings.

    Args:
        query_vec: Encoded query pattern
        kb_bundle: KB bundle for the predicate
        backend: Kernel backend for unbinding

    Returns:
        Unbounded vector containing variable bindings

    Example:
        >>> # Query: parent(alice, X) encoded as query_vec
        >>> # Bundle contains: parent(alice, bob) + parent(alice, carol) + ...
        >>> result = unbind_query_from_bundle(query_vec, kb_bundle, backend)
        >>> # Result contains: hv(ρ2 ⊗ bob) + hv(ρ2 ⊗ carol) + ... (noisy)
    """
    # Unbind query from bundle: bundle ⊗^(-1) query
    result = backend.unbind(kb_bundle, query_vec)
    return backend.normalize(result)


def unbind_role(
    vector: jnp.ndarray,
    role_vec: jnp.ndarray,
    backend: KernelBackend,
) -> jnp.ndarray:
    """
    Unbind role vector to extract entity.

    After unbinding the query pattern, we have role-filler pairs
    for the variable position. This unbinds the role to extract
    the entity hypervector.

    Args:
        vector: Vector containing role-filler binding (ρ ⊗ entity)
        role_vec: Role vector (ρ)
        backend: Kernel backend for unbinding

    Returns:
        Entity hypervector (noisy)

    Example:
        >>> # vector = hv(ρ2 ⊗ bob)
        >>> role2 = role_manager.get_role(2)
        >>> entity_vec = unbind_role(vector, role2, backend)
        >>> # entity_vec ≈ hv(bob) (with some noise)
    """
    # Unbind role: (ρ ⊗ entity) ⊗^(-1) ρ → entity
    result = backend.unbind(vector, role_vec)
    return backend.normalize(result)


def extract_variable_binding(
    kb_bundle: jnp.ndarray,
    query_vec: jnp.ndarray,
    role_vec: jnp.ndarray,
    backend: KernelBackend,
) -> jnp.ndarray:
    """
    Extract variable binding from KB bundle.

    This is a convenience function that combines unbinding the query
    pattern and unbinding the role to extract the entity hypervector.

    Args:
        kb_bundle: KB bundle for the predicate
        query_vec: Encoded query pattern (with bound arguments)
        role_vec: Role vector for the variable position
        backend: Kernel backend

    Returns:
        Entity hypervector for the variable (noisy)

    Example:
        >>> # Query: parent(alice, X)
        >>> # Extract X by unbinding query and role
        >>> entity_vec = extract_variable_binding(
        ...     kb_bundle, query_vec, role2_vec, backend
        ... )
        >>> # entity_vec ≈ hv(bob) + hv(carol) + ... (bundled, noisy)
    """
    # Step 1: Unbind query pattern from bundle
    unbounded = unbind_query_from_bundle(query_vec, kb_bundle, backend)

    # Step 2: Unbind role to get entity
    entity_vec = unbind_role(unbounded, role_vec, backend)

    return entity_vec
