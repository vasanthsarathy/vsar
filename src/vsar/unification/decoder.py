"""Structure-aware decoder for VSAR - unbind → typed cleanup."""

from typing import Optional
from dataclasses import dataclass
import jax.numpy as jnp

from ..kernel.base import KernelBackend
from ..symbols.registry import SymbolRegistry
from ..symbols.spaces import SymbolSpace
from .substitution import Term, Constant, Variable


@dataclass
class Atom:
    """A logical atom: predicate(arg1, arg2, ...)."""
    predicate: str
    args: list[Term]

    def __repr__(self) -> str:
        args_str = ", ".join(arg.name for arg in self.args)
        return f"{self.predicate}({args_str})"

    def is_ground(self) -> bool:
        """Check if atom contains no variables."""
        return all(isinstance(arg, Constant) for arg in self.args)

    def get_variables(self) -> set[str]:
        """Get all variable names in this atom."""
        return {arg.name for arg in self.args if isinstance(arg, Variable)}


class StructureDecoder:
    """
    Structure-aware decoder using unbind → typed cleanup.

    This decoder reverses the encoding process to extract structured
    information from hypervectors. It handles:
    - Ground atoms (only constants)
    - Atoms with variables (unbound positions)
    - Nested terms (future work)

    Args:
        backend: VSA backend for bind/unbind operations
        registry: Symbol registry with typed codebooks

    Example:
        >>> backend = FHRRBackend(dim=2048, seed=42)
        >>> registry = SymbolRegistry(dim=2048, seed=42)
        >>> decoder = StructureDecoder(backend, registry)
        >>>
        >>> # Encode an atom
        >>> vec = encoder.encode_atom(Atom("parent", [Constant("alice"), Variable("X")]))
        >>>
        >>> # Decode it
        >>> atom = decoder.decode_atom(vec, threshold=0.1)
        >>> print(atom)  # parent(alice, X)
    """

    def __init__(self, backend: KernelBackend, registry: SymbolRegistry):
        """Initialize the decoder.

        Args:
            backend: VSA backend for bind/unbind operations
            registry: Symbol registry with typed codebooks
        """
        self.backend = backend
        self.registry = registry

    def decode_atom(
        self,
        vec: jnp.ndarray,
        threshold: float = 0.1,
        max_arity: int = 5
    ) -> Optional[Atom]:
        """
        Decode an atom vector via unbind → cleanup.

        This reverses the encoding process:
        1. Unbind TAG_ATOM to get payload (P_p ⊕ args_bundle)
        2. Cleanup payload in PREDICATES space to get predicate
        3. For each argument position:
           - Unbind ARGᵢ role from payload
           - Try cleanup in ENTITIES space
           - If cleanup succeeds → Constant
           - If cleanup fails → Variable

        Args:
            vec: Atom vector to decode
            threshold: Minimum similarity threshold for cleanup (0-1 range)
            max_arity: Maximum arity to try (since we don't have signatures yet)

        Returns:
            Decoded Atom if successful, None if predicate unknown
        """
        # 1. Unbind TAG_ATOM to get (P_p ⊕ args_bundle)
        tag_atom = self.registry.get(SymbolSpace.TAGS, "ATOM")
        if tag_atom is None:
            return None

        payload = self.backend.unbind(vec, tag_atom)

        # 2. Cleanup payload to get predicate (P_p is in the bundle)
        pred_candidates = self.registry.cleanup(SymbolSpace.PREDICATES, payload, k=1)
        if not pred_candidates or pred_candidates[0][1] < threshold:
            return None  # UNKNOWN predicate

        predicate = pred_candidates[0][0]

        # 3. args_bundle is also in the payload (bundled with predicate)
        # We can directly unbind ARG roles from the payload
        args_bundle = payload

        # 4. Decode arguments
        # TODO: Get arity from signature when available (Phase 3+)
        args = []
        for i in range(max_arity):
            arg_role_name = f"ARG{i+1}"
            arg_role = self.registry.get(SymbolSpace.ARG_ROLES, arg_role_name)
            if arg_role is None:
                break  # No more argument roles registered

            # Unbind argument role
            arg_vec = self.backend.unbind(args_bundle, arg_role)

            # Try to cleanup in ENTITIES space
            const_candidates = self.registry.cleanup(SymbolSpace.ENTITIES, arg_vec, k=1)
            if const_candidates and const_candidates[0][1] >= threshold:
                # Successfully decoded as constant
                args.append(Constant(const_candidates[0][0]))
            else:
                # Could not decode - treat as variable
                # Stop here since we don't know actual arity
                # (If this is a real arg, it's a variable; if not, we're done)
                break

        # If we decoded at least one argument, return the atom
        if args:
            return Atom(predicate, args)
        else:
            # Nullary predicate or all variables
            return Atom(predicate, [])

    def decode_with_pattern(
        self,
        vec: jnp.ndarray,
        pattern: Atom,
        threshold: float = 0.1
    ) -> Optional[Atom]:
        """
        Decode an atom using a pattern to guide variable placement.

        This is useful when you know the structure (e.g., parent(alice, X))
        and want to decode only the variable positions.

        Args:
            vec: Atom vector to decode
            pattern: Pattern atom with Variables where values should be decoded
            threshold: Minimum similarity threshold

        Returns:
            Decoded atom matching the pattern, or None if mismatch
        """
        # First decode normally
        decoded = self.decode_atom(vec, threshold=threshold, max_arity=len(pattern.args))
        if decoded is None:
            return None

        # Check if predicate matches
        if decoded.predicate != pattern.predicate:
            return None

        # Check if arity matches
        if len(decoded.args) != len(pattern.args):
            return None

        # Build result using pattern as guide
        result_args = []
        for pattern_arg, decoded_arg in zip(pattern.args, decoded.args):
            if isinstance(pattern_arg, Variable):
                # Variable position - use decoded value
                result_args.append(decoded_arg)
            elif isinstance(pattern_arg, Constant):
                # Constant position - verify match
                if isinstance(decoded_arg, Constant) and decoded_arg.name == pattern_arg.name:
                    result_args.append(decoded_arg)
                else:
                    return None  # Mismatch!
            else:
                return None  # Unknown term type

        return Atom(pattern.predicate, result_args)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"StructureDecoder(backend={self.backend.__class__.__name__}, dim={self.registry.dim})"
