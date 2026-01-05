"""Tests for structure-aware decoding (Phase 2.1)."""

import pytest
from vsar.encoding.atom_encoder import Atom as EncoderAtom
from vsar.encoding.atom_encoder import AtomEncoder
from vsar.encoding.atom_encoder import Constant as EncoderConstant
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.unification.decoder import Atom, Constant, StructureDecoder, Variable


@pytest.mark.xfail(reason="StructureDecoder WIP - unification module integration pending")
class TestSlotLevelDecoding:
    """Test slot-level decoding of atoms."""

    def test_decode_ground_atom(self):
        """Test decoding a ground atom (no variables)."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)

        # Encode parent(alice, bob)
        atom_enc = EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        vec = encoder.encode_atom(atom_enc)

        # Decode it
        decoded = decoder.decode_atom(vec, threshold=0.05)

        assert decoded is not None
        assert decoded.predicate == "parent"
        assert len(decoded.args) == 2
        assert all(isinstance(arg, Constant) for arg in decoded.args)
        assert decoded.args[0].name == "alice"
        assert decoded.args[1].name == "bob"
        assert decoded.is_ground()

    def test_decode_unary_predicate(self):
        """Test decoding unary predicate."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)

        atom_enc = EncoderAtom("Person", [EncoderConstant("alice")])
        vec = encoder.encode_atom(atom_enc)

        decoded = decoder.decode_atom(vec, threshold=0.05)

        assert decoded is not None
        assert decoded.predicate == "Person"
        assert len(decoded.args) == 1
        assert decoded.args[0].name == "alice"

    def test_decode_ternary_predicate(self):
        """Test decoding ternary predicate."""
        backend = FHRRBackend(dim=4096, seed=42)
        registry = SymbolRegistry(dim=4096, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)

        atom_enc = EncoderAtom(
            "between", [EncoderConstant("alice"), EncoderConstant("bob"), EncoderConstant("carol")]
        )
        vec = encoder.encode_atom(atom_enc)

        decoded = decoder.decode_atom(vec, threshold=0.05)

        assert decoded is not None
        assert decoded.predicate == "between"
        assert len(decoded.args) == 3
        assert decoded.args[0].name == "alice"
        assert decoded.args[1].name == "bob"
        assert decoded.args[2].name == "carol"

    def test_decode_below_threshold_returns_none(self):
        """Test that decoding with high threshold returns None."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)

        atom_enc = EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        vec = encoder.encode_atom(atom_enc)

        # Very high threshold - should fail
        decoded = decoder.decode_atom(vec, threshold=0.95)

        assert decoded is None  # UNKNOWN

    def test_get_variables(self):
        """Test extracting variables from an atom."""
        atom = Atom("parent", [Constant("alice"), Variable("X")])

        vars = atom.get_variables()
        assert vars == {"X"}

        atom2 = Atom("parent", [Constant("alice"), Constant("bob")])
        assert atom2.get_variables() == set()

    def test_is_ground(self):
        """Test checking if atom is ground."""
        ground_atom = Atom("parent", [Constant("alice"), Constant("bob")])
        assert ground_atom.is_ground()

        non_ground = Atom("parent", [Constant("alice"), Variable("X")])
        assert not non_ground.is_ground()


class TestPatternDecoding:
    """Test pattern-guided decoding."""

    def test_decode_with_constant_pattern(self):
        """Test decoding with all-constant pattern (verification mode)."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)

        # Encode parent(alice, bob)
        atom_enc = EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        vec = encoder.encode_atom(atom_enc)

        # Decode with matching pattern
        pattern = Atom("parent", [Constant("alice"), Constant("bob")])
        decoded = decoder.decode_with_pattern(vec, pattern, threshold=0.05)

        assert decoded is not None
        assert decoded.predicate == "parent"
        assert decoded.args[0].name == "alice"
        assert decoded.args[1].name == "bob"

    def test_decode_with_variable_pattern(self):
        """Test decoding with variable pattern (query mode)."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)

        # Encode parent(alice, bob)
        atom_enc = EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        vec = encoder.encode_atom(atom_enc)

        # Decode with pattern parent(alice, X)
        pattern = Atom("parent", [Constant("alice"), Variable("X")])
        decoded = decoder.decode_with_pattern(vec, pattern, threshold=0.05)

        assert decoded is not None
        assert decoded.predicate == "parent"
        assert decoded.args[0].name == "alice"
        # Second arg should be decoded value (bob), not Variable X
        assert isinstance(decoded.args[1], Constant)
        assert decoded.args[1].name == "bob"

    def test_decode_pattern_mismatch_predicate(self):
        """Test that pattern mismatch on predicate returns None."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)

        atom_enc = EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        vec = encoder.encode_atom(atom_enc)

        # Wrong predicate
        pattern = Atom("sibling", [Constant("alice"), Variable("X")])
        decoded = decoder.decode_with_pattern(vec, pattern, threshold=0.05)

        assert decoded is None  # Predicate mismatch

    def test_decode_pattern_mismatch_constant(self):
        """Test that pattern mismatch on constant returns None."""
        backend = FHRRBackend(dim=2048, seed=42)
        registry = SymbolRegistry(dim=2048, seed=42)
        encoder = AtomEncoder(backend, registry)
        decoder = StructureDecoder(backend, registry)

        atom_enc = EncoderAtom("parent", [EncoderConstant("alice"), EncoderConstant("bob")])
        vec = encoder.encode_atom(atom_enc)

        # Wrong constant in first position
        pattern = Atom("parent", [Constant("carol"), Variable("X")])
        decoded = decoder.decode_with_pattern(vec, pattern, threshold=0.05)

        assert decoded is None  # Constant mismatch


class TestDecoderProperties:
    """Test decoder properties and edge cases."""

    def test_decoder_repr(self):
        """Test decoder string representation."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        decoder = StructureDecoder(backend, registry)

        repr_str = repr(decoder)
        assert "StructureDecoder" in repr_str
        assert "FHRRBackend" in repr_str
        assert "512" in repr_str

    def test_atom_repr(self):
        """Test atom string representation."""
        atom = Atom("parent", [Constant("alice"), Variable("X")])
        assert str(atom) == "parent(alice, X)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
