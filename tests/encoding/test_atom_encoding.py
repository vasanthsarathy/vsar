"""Tests for atom encoding with bind operations (Phase 1.2).

This test suite validates:
- Encoding atoms using bind: enc(p(t1,...,tk)) = (P_p ⊗ TAG_ATOM) ⊗ (⊕ᵢ ARGᵢ ⊗ enc(tᵢ))
- Decoding atoms using unbind → cleanup
- Handling nested terms
- Variable representation
"""

import jax
import jax.numpy as jnp
import pytest

from vsar.symbols.spaces import SymbolSpace
from vsar.symbols.registry import SymbolRegistry
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.encoding.atom_encoder import AtomEncoder, Atom, Constant, Term


class TestAtomEncoding:
    """Test encoding atoms with bind operations."""

    def test_encode_simple_atom(self):
        """Test encoding a simple ground atom: parent(alice, bob)."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        encoder = AtomEncoder(backend, registry)

        # Encode parent(alice, bob)
        atom = Atom("parent", [Constant("alice"), Constant("bob")])
        vec = encoder.encode_atom(atom)

        assert vec.shape == (512,)
        assert jnp.iscomplexobj(vec)
        # Vector should be normalized
        assert jnp.abs(jnp.linalg.norm(vec) - 1.0) < 0.01

    def test_encode_different_atoms_are_distinct(self):
        """Test that different atoms encode to different vectors."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        encoder = AtomEncoder(backend, registry)

        atom1 = Atom("parent", [Constant("alice"), Constant("bob")])
        atom2 = Atom("parent", [Constant("alice"), Constant("carol")])
        atom3 = Atom("sibling", [Constant("bob"), Constant("carol")])

        vec1 = encoder.encode_atom(atom1)
        vec2 = encoder.encode_atom(atom2)
        vec3 = encoder.encode_atom(atom3)

        # Different atoms should have low similarity
        sim_12 = backend.similarity(vec1, vec2)
        sim_13 = backend.similarity(vec1, vec3)
        sim_23 = backend.similarity(vec2, vec3)

        assert sim_12 < 0.85  # Different arguments (same predicate)
        assert sim_13 < 0.6  # Different predicates
        assert sim_23 < 0.85  # Different predicates and arguments

    def test_encode_same_atom_is_identical(self):
        """Test that encoding the same atom twice gives the same result."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        encoder = AtomEncoder(backend, registry)

        atom = Atom("parent", [Constant("alice"), Constant("bob")])
        vec1 = encoder.encode_atom(atom)
        vec2 = encoder.encode_atom(atom)

        # Same atom should encode identically
        assert jnp.allclose(vec1, vec2)


class TestAtomDecoding:
    """Test decoding atoms via unbind → cleanup."""

    def test_decode_predicate(self):
        """Test decoding predicate from atom encoding."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        encoder = AtomEncoder(backend, registry)

        atom = Atom("parent", [Constant("alice"), Constant("bob")])
        vec = encoder.encode_atom(atom)

        # Decode predicate: enc ⊘ TAG_ATOM → cleanup to predicate space
        tag_atom = registry.get(SymbolSpace.TAGS, "ATOM")
        payload = backend.unbind(vec, tag_atom)

        # Cleanup in PREDICATES space
        results = registry.cleanup(SymbolSpace.PREDICATES, payload, k=1)

        assert len(results) == 1
        assert results[0][0] == "parent"
        assert results[0][1] > 0.6  # Good similarity despite bundling

    def test_decode_arg_positions(self):
        """Test decoding specific argument positions."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)

        atom = Atom("parent", [Constant("alice"), Constant("bob")])
        vec = encoder.encode_atom(atom)

        # Step 1: Unbind TAG_ATOM to get (P_p ⊕ args_bundle)
        tag_atom = registry.get(SymbolSpace.TAGS, "ATOM")
        payload = backend.unbind(vec, tag_atom)

        # Step 2: args_bundle is in the payload (bundled with predicate)
        # We unbind ARG roles directly from payload
        args_bundle = payload

        # Step 3: Unbind ARG₁
        arg1_role = registry.get(SymbolSpace.ARG_ROLES, "ARG1")
        arg1_vec = backend.unbind(args_bundle, arg1_role)

        # Cleanup in ENTITIES space
        results1 = registry.cleanup(SymbolSpace.ENTITIES, arg1_vec, k=1)
        assert len(results1) == 1
        assert results1[0][0] == "alice"
        assert results1[0][1] > 0.05  # Very approximate due to bundling noise

        # Step 4: Unbind ARG₂
        arg2_role = registry.get(SymbolSpace.ARG_ROLES, "ARG2")
        arg2_vec = backend.unbind(args_bundle, arg2_role)

        results2 = registry.cleanup(SymbolSpace.ENTITIES, arg2_vec, k=1)
        assert len(results2) == 1
        assert results2[0][0] == "bob"
        assert results2[0][1] > 0.05  # Very approximate due to bundling noise

    def test_decode_full_atom(self):
        """Test decoding complete atom structure."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)

        atom = Atom("parent", [Constant("alice"), Constant("bob")])
        vec = encoder.encode_atom(atom)

        # Decode the atom with low threshold for approximate VSA recovery
        decoded = encoder.decode_atom(vec, threshold=0.05)

        assert decoded is not None
        assert decoded.predicate == "parent"
        assert len(decoded.args) == 2
        assert decoded.args[0].name == "alice"
        assert decoded.args[1].name == "bob"


class TestUnaryPredicates:
    """Test unary predicates (concepts)."""

    def test_encode_unary_predicate(self):
        """Test encoding unary predicate: Person(alice)."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        encoder = AtomEncoder(backend, registry)

        atom = Atom("Person", [Constant("alice")])
        vec = encoder.encode_atom(atom)

        assert vec.shape == (512,)

    def test_decode_unary_predicate(self):
        """Test decoding unary predicate."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)

        atom = Atom("Person", [Constant("alice")])
        vec = encoder.encode_atom(atom)

        decoded = encoder.decode_atom(vec, threshold=0.05)

        assert decoded is not None
        assert decoded.predicate == "Person"
        assert len(decoded.args) == 1
        assert decoded.args[0].name == "alice"


class TestHigherArityPredicates:
    """Test predicates with arity > 2."""

    def test_encode_ternary_predicate(self):
        """Test encoding ternary predicate: between(a, b, c)."""
        backend = FHRRBackend(dim=1024, seed=42)
        registry = SymbolRegistry(dim=1024, seed=42)
        encoder = AtomEncoder(backend, registry)

        atom = Atom("between", [
            Constant("alice"),
            Constant("bob"),
            Constant("carol")
        ])
        vec = encoder.encode_atom(atom)

        assert vec.shape == (1024,)

    def test_decode_ternary_predicate(self):
        """Test decoding ternary predicate."""
        backend = FHRRBackend(dim=4096, seed=42)
        registry = SymbolRegistry(dim=4096, seed=42)
        encoder = AtomEncoder(backend, registry)

        atom = Atom("between", [
            Constant("alice"),
            Constant("bob"),
            Constant("carol")
        ])
        vec = encoder.encode_atom(atom)

        decoded = encoder.decode_atom(vec, threshold=0.05)

        assert decoded is not None
        assert decoded.predicate == "between"
        assert len(decoded.args) == 3
        assert decoded.args[0].name == "alice"
        assert decoded.args[1].name == "bob"
        assert decoded.args[2].name == "carol"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
