"""Atom encoder using FHRR bind operations for VSAR v2.0.

This module implements atom encoding using the formula:
enc(p(t1,...,tk)) = (P_p ⊗ TAG_ATOM) ⊗ (⊕ᵢ ARGᵢ ⊗ enc(tᵢ))

Where:
- P_p is the predicate vector from PREDICATES space
- TAG_ATOM is the type tag from TAGS space
- ARGᵢ are argument role markers from ARG_ROLES space
- tᵢ are term encodings (constants from ENTITIES space)
"""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp

from ..kernel.base import KernelBackend
from ..symbols.registry import SymbolRegistry
from ..symbols.spaces import SymbolSpace


@dataclass
class Term:
    """Base class for logical terms."""
    name: str


@dataclass
class Constant(Term):
    """A constant term (e.g., alice, bob)."""
    pass


@dataclass
class Variable(Term):
    """A variable term (e.g., X, Y)."""
    pass


@dataclass
class Atom:
    """A logical atom: predicate(arg1, arg2, ...)."""
    predicate: str
    args: list[Term]

    def __repr__(self) -> str:
        args_str = ", ".join(arg.name for arg in self.args)
        return f"{self.predicate}({args_str})"


class AtomEncoder:
    """
    Encoder for logical atoms using FHRR bind operations.

    This class implements the structure-aware encoding scheme from the
    VSAR v2.0 specification, where atoms are encoded as:

    enc(p(t1,...,tk)) = (P_p ⊗ TAG_ATOM) ⊗ (⊕ᵢ ARGᵢ ⊗ enc(tᵢ))

    Args:
        backend: VSA backend for bind/unbind operations
        registry: Symbol registry with typed codebooks

    Example:
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> registry = SymbolRegistry(dim=512, seed=42)
        >>> encoder = AtomEncoder(backend, registry)
        >>>
        >>> atom = Atom("parent", [Constant("alice"), Constant("bob")])
        >>> vec = encoder.encode_atom(atom)
        >>> decoded = encoder.decode_atom(vec)
        >>> print(decoded)  # parent(alice, bob)
    """

    def __init__(self, backend: KernelBackend, registry: SymbolRegistry):
        """Initialize the atom encoder.

        Args:
            backend: VSA backend for bind/unbind operations
            registry: Symbol registry with typed codebooks
        """
        self.backend = backend
        self.registry = registry

        # Ensure TAG_ATOM is registered
        self.registry.register(SymbolSpace.TAGS, "ATOM")

    def encode_term(self, term: Term) -> jnp.ndarray:
        """
        Encode a term (constant or variable).

        For constants, returns the vector from ENTITIES space.
        For variables, returns a placeholder (not yet implemented).

        Args:
            term: The term to encode

        Returns:
            Hypervector encoding of the term
        """
        if isinstance(term, Constant):
            return self.registry.register(SymbolSpace.ENTITIES, term.name)
        elif isinstance(term, Variable):
            # TODO: Implement variable encoding (Phase 2+)
            raise NotImplementedError("Variable encoding not yet implemented")
        else:
            raise ValueError(f"Unknown term type: {type(term)}")

    def encode_atom(self, atom: Atom) -> jnp.ndarray:
        """
        Encode a logical atom using bind operations.

        Formula: enc(p(t1,...,tk)) = (P_p ⊗ TAG_ATOM) ⊗ (⊕ᵢ ARGᵢ ⊗ enc(tᵢ))

        Steps:
        1. Get predicate vector P_p from PREDICATES space
        2. Get TAG_ATOM from TAGS space
        3. For each argument:
           - Get ARGᵢ role marker
           - Encode the term
           - Bind role with term: ARGᵢ ⊗ enc(tᵢ)
        4. Bundle all argument bindings: ⊕ᵢ (ARGᵢ ⊗ enc(tᵢ))
        5. Bind predicate+tag with args: (P_p ⊗ TAG_ATOM) ⊗ args_bundle

        Args:
            atom: The atom to encode

        Returns:
            Hypervector encoding of the atom
        """
        # 1. Get predicate vector
        pred_vec = self.registry.register(SymbolSpace.PREDICATES, atom.predicate)

        # 2. Get TAG_ATOM
        tag_atom = self.registry.get(SymbolSpace.TAGS, "ATOM")

        # 3. Encode each argument with its role
        arg_bundles = []
        for i, arg in enumerate(atom.args):
            # Get argument role marker (ARG1, ARG2, ...)
            arg_role_name = f"ARG{i+1}"
            arg_role = self.registry.register(SymbolSpace.ARG_ROLES, arg_role_name)

            # Encode the term
            arg_vec = self.encode_term(arg)

            # Bind role with term
            bound_arg = self.backend.bind(arg_role, arg_vec)
            arg_bundles.append(bound_arg)

        # 4. Bundle all arguments
        if len(arg_bundles) == 1:
            args_bundle = arg_bundles[0]
        else:
            args_bundle = self.backend.bundle(arg_bundles)

        # 5. Create atom encoding using bundling for easier cleanup
        # enc = TAG_ATOM ⊗ (P_p ⊕ args_bundle)
        # This allows both predicate and args to be extracted via cleanup
        pred_and_args = self.backend.bundle([pred_vec, args_bundle])
        result = self.backend.bind(tag_atom, pred_and_args)

        return self.backend.normalize(result)

    def decode_atom(
        self,
        vec: jnp.ndarray,
        threshold: float = 0.25
    ) -> Optional[Atom]:
        """
        Decode an atom vector via unbind → cleanup.

        This reverses the encoding process:
        1. Unbind TAG_ATOM to get payload
        2. Cleanup payload in PREDICATES space to get predicate
        3. Unbind predicate to get args bundle
        4. For each argument position:
           - Unbind ARGᵢ role
           - Cleanup in ENTITIES space to get constant

        Returns None if any cleanup is below threshold (UNKNOWN).

        Args:
            vec: Atom vector to decode
            threshold: Minimum similarity threshold for cleanup (0-1 range)

        Returns:
            Decoded Atom if successful, None if below threshold
        """
        # 1. Unbind TAG_ATOM to get (P_p ⊕ args_bundle)
        tag_atom = self.registry.get(SymbolSpace.TAGS, "ATOM")
        if tag_atom is None:
            return None

        payload = self.backend.unbind(vec, tag_atom)

        # 2. Cleanup payload to get predicate (P_p is in the bundle)
        pred_candidates = self.registry.cleanup(SymbolSpace.PREDICATES, payload, k=1)
        if not pred_candidates or pred_candidates[0][1] < threshold:
            return None  # UNKNOWN

        predicate = pred_candidates[0][0]

        # 3. args_bundle is also in the payload (bundled with predicate)
        # We can directly unbind ARG roles from the payload
        args_bundle = payload

        # 4. Decode arguments
        # For now, we'll try to decode up to 5 argument positions
        # TODO: Get arity from signature (Phase 2+)
        args = []
        for i in range(5):  # Max arity for now
            arg_role_name = f"ARG{i+1}"
            arg_role = self.registry.get(SymbolSpace.ARG_ROLES, arg_role_name)
            if arg_role is None:
                break  # No more argument roles registered

            # Unbind argument role
            arg_vec = self.backend.unbind(args_bundle, arg_role)

            # Try to cleanup in ENTITIES space
            const_candidates = self.registry.cleanup(SymbolSpace.ENTITIES, arg_vec, k=1)
            if const_candidates and const_candidates[0][1] >= threshold:
                args.append(Constant(const_candidates[0][0]))
            else:
                # If we can't decode an argument that we expect, stop
                # (We don't know the actual arity without a signature)
                break

        # If we decoded at least one argument, return the atom
        if args:
            return Atom(predicate, args)
        else:
            return None

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"AtomEncoder(backend={self.backend.__class__.__name__}, dim={self.registry.dim})"
