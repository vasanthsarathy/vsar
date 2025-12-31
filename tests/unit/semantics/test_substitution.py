"""Tests for variable substitution."""

import pytest

from vsar.language.ast import Atom
from vsar.semantics.substitution import (
    Substitution,
    get_atom_unique_variables,
    get_atom_variables,
    is_variable,
)


class TestSubstitution:
    """Test Substitution class."""

    def test_empty_substitution(self) -> None:
        """Test creating empty substitution."""
        sub = Substitution()
        assert sub.is_empty()
        assert sub.vars() == []

    def test_bind_single_variable(self) -> None:
        """Test binding a single variable."""
        sub = Substitution()
        sub2 = sub.bind("X", "alice")

        assert sub2.has("X")
        assert sub2.get("X") == "alice"
        assert not sub2.is_empty()

    def test_bind_multiple_variables(self) -> None:
        """Test binding multiple variables."""
        sub = Substitution()
        sub2 = sub.bind("X", "alice").bind("Y", "bob")

        assert sub2.get("X") == "alice"
        assert sub2.get("Y") == "bob"
        assert set(sub2.vars()) == {"X", "Y"}

    def test_bind_is_immutable(self) -> None:
        """Test that bind returns a new substitution."""
        sub1 = Substitution()
        sub2 = sub1.bind("X", "alice")

        # sub1 should be unchanged
        assert sub1.is_empty()
        assert not sub2.is_empty()

    def test_get_unbound_variable(self) -> None:
        """Test getting an unbound variable returns None."""
        sub = Substitution()
        assert sub.get("X") is None

    def test_has_variable(self) -> None:
        """Test checking if variable is bound."""
        sub = Substitution().bind("X", "alice")
        assert sub.has("X")
        assert not sub.has("Y")

    def test_apply_to_ground_atom(self) -> None:
        """Test applying substitution to ground atom (no variables)."""
        sub = Substitution().bind("X", "alice")
        atom = Atom(predicate="parent", args=["alice", "bob"])
        result = sub.apply_to_atom(atom)

        assert result.predicate == "parent"
        assert result.args == ["alice", "bob"]

    def test_apply_to_atom_with_variables(self) -> None:
        """Test applying substitution to atom with variables."""
        sub = Substitution().bind("X", "alice").bind("Y", "bob")
        atom = Atom(predicate="parent", args=["X", "Y"])
        result = sub.apply_to_atom(atom)

        assert result.predicate == "parent"
        assert result.args == ["alice", "bob"]
        assert result.is_ground()

    def test_apply_partial_substitution(self) -> None:
        """Test applying substitution that doesn't bind all variables."""
        sub = Substitution().bind("X", "alice")
        atom = Atom(predicate="parent", args=["X", "Y"])
        result = sub.apply_to_atom(atom)

        assert result.args == ["alice", "Y"]
        assert not result.is_ground()
        assert result.get_variables() == ["Y"]

    def test_apply_mixed_args(self) -> None:
        """Test applying substitution to atom with mixed constants and variables."""
        sub = Substitution().bind("X", "alice")
        atom = Atom(predicate="transfer", args=["X", "bob", "money"])
        result = sub.apply_to_atom(atom)

        assert result.args == ["alice", "bob", "money"]

    def test_compose_empty_substitutions(self) -> None:
        """Test composing two empty substitutions."""
        sub1 = Substitution()
        sub2 = Substitution()
        result = sub1.compose(sub2)

        assert result.is_empty()

    def test_compose_substitutions(self) -> None:
        """Test composing two substitutions."""
        sub1 = Substitution().bind("X", "alice")
        sub2 = Substitution().bind("Y", "bob")
        result = sub1.compose(sub2)

        assert result.get("X") == "alice"
        assert result.get("Y") == "bob"
        assert set(result.vars()) == {"X", "Y"}

    def test_compose_with_conflict(self) -> None:
        """Test composing substitutions with conflicting bindings."""
        sub1 = Substitution().bind("X", "alice")
        sub2 = Substitution().bind("X", "bob")
        result = sub1.compose(sub2)

        # Other's binding takes precedence
        assert result.get("X") == "bob"

    def test_repr_empty(self) -> None:
        """Test string representation of empty substitution."""
        sub = Substitution()
        assert repr(sub) == "Substitution({})"

    def test_repr_with_bindings(self) -> None:
        """Test string representation with bindings."""
        sub = Substitution().bind("X", "alice").bind("Y", "bob")
        # Should be sorted alphabetically
        assert repr(sub) == "Substitution({X=alice, Y=bob})"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_variable_uppercase(self) -> None:
        """Test that uppercase terms are variables."""
        assert is_variable("X")
        assert is_variable("Y")
        assert is_variable("Person")
        assert is_variable("VariableName")

    def test_is_variable_lowercase(self) -> None:
        """Test that lowercase terms are not variables."""
        assert not is_variable("alice")
        assert not is_variable("bob")
        assert not is_variable("constant")

    def test_is_variable_empty(self) -> None:
        """Test that empty string is not a variable."""
        assert not is_variable("")

    def test_get_atom_variables(self) -> None:
        """Test getting variables from atom."""
        atom = Atom(predicate="parent", args=["X", "bob", "Y"])
        variables = get_atom_variables(atom)

        assert variables == ["X", "Y"]

    def test_get_atom_variables_no_variables(self) -> None:
        """Test getting variables from ground atom."""
        atom = Atom(predicate="parent", args=["alice", "bob"])
        variables = get_atom_variables(atom)

        assert variables == []

    def test_get_atom_variables_all_variables(self) -> None:
        """Test getting variables when all args are variables."""
        atom = Atom(predicate="test", args=["X", "Y", "Z"])
        variables = get_atom_variables(atom)

        assert variables == ["X", "Y", "Z"]

    def test_get_atom_variables_duplicates(self) -> None:
        """Test that get_atom_variables includes duplicates."""
        atom = Atom(predicate="same", args=["X", "X"])
        variables = get_atom_variables(atom)

        assert variables == ["X", "X"]

    def test_get_atom_unique_variables(self) -> None:
        """Test getting unique variables from atom."""
        atom = Atom(predicate="same", args=["X", "X", "Y"])
        variables = get_atom_unique_variables(atom)

        assert variables == ["X", "Y"]

    def test_get_atom_unique_variables_preserves_order(self) -> None:
        """Test that unique variables preserves first occurrence order."""
        atom = Atom(predicate="test", args=["Y", "X", "Y", "Z", "X"])
        variables = get_atom_unique_variables(atom)

        assert variables == ["Y", "X", "Z"]


class TestSubstitutionIntegration:
    """Integration tests for substitution with atoms."""

    def test_apply_and_check_ground(self) -> None:
        """Test applying substitution and checking if result is ground."""
        sub = Substitution().bind("X", "alice").bind("Y", "bob")
        atom = Atom(predicate="parent", args=["X", "Y"])

        result = sub.apply_to_atom(atom)
        assert result.is_ground()
        assert result.get_variables() == []

    def test_chain_substitutions(self) -> None:
        """Test chaining multiple substitutions."""
        atom = Atom(predicate="test", args=["X", "Y", "Z"])

        sub1 = Substitution().bind("X", "a")
        result1 = sub1.apply_to_atom(atom)
        assert result1.args == ["a", "Y", "Z"]

        sub2 = Substitution().bind("Y", "b")
        result2 = sub2.apply_to_atom(result1)
        assert result2.args == ["a", "b", "Z"]

        sub3 = Substitution().bind("Z", "c")
        result3 = sub3.apply_to_atom(result2)
        assert result3.args == ["a", "b", "c"]
        assert result3.is_ground()

    def test_compose_and_apply(self) -> None:
        """Test composing substitutions and applying the result."""
        sub1 = Substitution().bind("X", "alice")
        sub2 = Substitution().bind("Y", "bob")
        combined = sub1.compose(sub2)

        atom = Atom(predicate="parent", args=["X", "Y"])
        result = combined.apply_to_atom(atom)

        assert result.args == ["alice", "bob"]
        assert result.is_ground()
