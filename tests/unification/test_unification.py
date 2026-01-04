"""Tests for unification via decoding (Phase 2.2)."""

import pytest
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.unification.decoder import Atom, Constant, Variable
from vsar.unification.substitution import Substitution
from vsar.unification.unifier import Unifier


class TestGroundUnification:
    """Test unification of ground atoms (no variables)."""

    def test_unify_identical_atoms(self):
        """Test: unify(parent(alice,bob), parent(alice,bob)) → {}"""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        unifier = Unifier(backend, registry)

        atom1 = Atom("parent", [Constant("alice"), Constant("bob")])
        atom2 = Atom("parent", [Constant("alice"), Constant("bob")])

        subst = unifier.unify(atom1, atom2)

        assert subst is not None
        assert subst.is_empty()
        assert len(subst) == 0

    def test_unify_different_predicates_fails(self):
        """Test: unify(parent(alice,bob), sibling(alice,bob)) → None"""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        unifier = Unifier(backend, registry)

        atom1 = Atom("parent", [Constant("alice"), Constant("bob")])
        atom2 = Atom("sibling", [Constant("alice"), Constant("bob")])

        subst = unifier.unify(atom1, atom2)

        assert subst is None  # Different predicates

    def test_unify_different_constants_fails(self):
        """Test: unify(parent(alice,bob), parent(carol,bob)) → None"""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        unifier = Unifier(backend, registry)

        atom1 = Atom("parent", [Constant("alice"), Constant("bob")])
        atom2 = Atom("parent", [Constant("carol"), Constant("bob")])

        subst = unifier.unify(atom1, atom2)

        assert subst is None  # Different constants

    def test_unify_different_arity_fails(self):
        """Test that different arity atoms don't unify."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        unifier = Unifier(backend, registry)

        atom1 = Atom("parent", [Constant("alice"), Constant("bob")])
        atom2 = Atom("parent", [Constant("alice")])

        subst = unifier.unify(atom1, atom2)

        assert subst is None  # Different arity


class TestVariableUnification:
    """Test unification with variables."""

    def test_unify_variable_with_constant(self):
        """Test: unify(parent(alice,X), parent(alice,bob)) → {X: bob}"""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        unifier = Unifier(backend, registry)

        atom1 = Atom("parent", [Constant("alice"), Variable("X")])
        atom2 = Atom("parent", [Constant("alice"), Constant("bob")])

        subst = unifier.unify(atom1, atom2)

        assert subst is not None
        assert len(subst) == 1
        assert "X" in subst
        assert subst.get("X").name == "bob"

    def test_unify_multiple_variables(self):
        """Test: unify(parent(X,Y), parent(alice,bob)) → {X: alice, Y: bob}"""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        unifier = Unifier(backend, registry)

        atom1 = Atom("parent", [Variable("X"), Variable("Y")])
        atom2 = Atom("parent", [Constant("alice"), Constant("bob")])

        subst = unifier.unify(atom1, atom2)

        assert subst is not None
        assert len(subst) == 2
        assert subst.get("X").name == "alice"
        assert subst.get("Y").name == "bob"

    def test_unify_variable_with_variable(self):
        """Test: unify(parent(X,Y), parent(A,B)) → {X: A, Y: B}"""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        unifier = Unifier(backend, registry)

        atom1 = Atom("parent", [Variable("X"), Variable("Y")])
        atom2 = Atom("parent", [Variable("A"), Variable("B")])

        subst = unifier.unify(atom1, atom2)

        assert subst is not None
        assert len(subst) == 2
        # Variables get bound to each other
        assert "X" in subst
        assert "Y" in subst

    def test_unify_same_variable_twice(self):
        """Test: unify(parent(X,X), parent(alice,alice)) → {X: alice}"""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        unifier = Unifier(backend, registry)

        atom1 = Atom("parent", [Variable("X"), Variable("X")])
        atom2 = Atom("parent", [Constant("alice"), Constant("alice")])

        subst = unifier.unify(atom1, atom2)

        assert subst is not None
        assert len(subst) == 1
        assert subst.get("X").name == "alice"

    def test_unify_same_variable_conflict_fails(self):
        """Test: unify(parent(X,X), parent(alice,bob)) → None"""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        unifier = Unifier(backend, registry)

        atom1 = Atom("parent", [Variable("X"), Variable("X")])
        atom2 = Atom("parent", [Constant("alice"), Constant("bob")])

        subst = unifier.unify(atom1, atom2)

        assert subst is None  # X can't be both alice and bob


class TestSubstitutionOperations:
    """Test substitution operations."""

    def test_substitution_empty(self):
        """Test empty substitution."""
        subst = Substitution()

        assert subst.is_empty()
        assert len(subst) == 0
        assert str(subst) == "{}"

    def test_substitution_bind(self):
        """Test binding variables."""
        subst = Substitution()

        assert subst.bind("X", Constant("alice"))
        assert len(subst) == 1
        assert subst.get("X").name == "alice"

    def test_substitution_bind_conflict(self):
        """Test that conflicting bindings fail."""
        subst = Substitution()
        subst.bind("X", Constant("alice"))

        # Try to bind X to different value
        result = subst.bind("X", Constant("bob"))

        assert not result  # Conflict!
        assert subst.get("X").name == "alice"  # Original binding preserved

    def test_substitution_compose(self):
        """Test composition of substitutions."""
        subst1 = Substitution({"X": Constant("alice")})
        subst2 = Substitution({"Y": Constant("bob")})

        composed = subst1.compose(subst2)

        assert len(composed) == 2
        assert composed.get("X").name == "alice"
        assert composed.get("Y").name == "bob"

    def test_substitution_compose_override(self):
        """Test that composition applies substitutions."""
        subst1 = Substitution({"X": Variable("Y")})
        subst2 = Substitution({"Y": Constant("alice")})

        composed = subst1.compose(subst2)

        # X should be bound to alice (through Y)
        assert composed.get("X").name == "alice"

    def test_substitution_apply_to_variable(self):
        """Test applying substitution to variable."""
        subst = Substitution({"X": Constant("alice")})

        var = Variable("X")
        result = var.apply_substitution(subst)

        assert isinstance(result, Constant)
        assert result.name == "alice"

    def test_substitution_apply_to_unbound_variable(self):
        """Test applying substitution to unbound variable."""
        subst = Substitution({"X": Constant("alice")})

        var = Variable("Y")  # Not in substitution
        result = var.apply_substitution(subst)

        assert isinstance(result, Variable)
        assert result.name == "Y"

    def test_substitution_repr(self):
        """Test substitution string representation."""
        subst = Substitution({"X": Constant("alice"), "Y": Constant("bob")})

        repr_str = str(subst)
        assert "X" in repr_str
        assert "alice" in repr_str
        assert "Y" in repr_str
        assert "bob" in repr_str

    def test_substitution_equality(self):
        """Test substitution equality."""
        subst1 = Substitution({"X": Constant("alice")})
        subst2 = Substitution({"X": Constant("alice")})
        subst3 = Substitution({"X": Constant("bob")})

        assert subst1 == subst2
        assert subst1 != subst3


class TestUnifierProperties:
    """Test unifier properties."""

    def test_unifier_repr(self):
        """Test unifier string representation."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=512, seed=42)
        unifier = Unifier(backend, registry)

        repr_str = repr(unifier)
        assert "Unifier" in repr_str
        assert "FHRRBackend" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
