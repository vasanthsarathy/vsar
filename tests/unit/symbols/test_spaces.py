"""Unit tests for symbol spaces."""


import pytest

from vsar.symbols.spaces import SymbolSpace


class TestSymbolSpace:
    """Test cases for SymbolSpace enum."""

    def test_all_spaces_defined(self) -> None:
        """Test that all expected symbol spaces are defined."""
        assert SymbolSpace.ENTITIES
        assert SymbolSpace.RELATIONS
        assert SymbolSpace.ATTRIBUTES
        assert SymbolSpace.CONTEXTS
        assert SymbolSpace.TIME
        assert SymbolSpace.STRUCTURAL

    def test_space_values(self) -> None:
        """Test that spaces have correct abbreviations."""
        assert SymbolSpace.ENTITIES.value == "E"
        assert SymbolSpace.RELATIONS.value == "R"
        assert SymbolSpace.ATTRIBUTES.value == "A"
        assert SymbolSpace.CONTEXTS.value == "C"
        assert SymbolSpace.TIME.value == "T"
        assert SymbolSpace.STRUCTURAL.value == "S"

    def test_str_representation(self) -> None:
        """Test string representation returns abbreviation."""
        assert str(SymbolSpace.ENTITIES) == "E"
        assert str(SymbolSpace.RELATIONS) == "R"

    def test_repr_representation(self) -> None:
        """Test repr returns full enum name."""
        assert repr(SymbolSpace.ENTITIES) == "SymbolSpace.ENTITIES"
        assert repr(SymbolSpace.RELATIONS) == "SymbolSpace.RELATIONS"

    def test_enum_equality(self) -> None:
        """Test enum equality comparisons."""
        assert SymbolSpace.ENTITIES == SymbolSpace.ENTITIES
        assert SymbolSpace.ENTITIES != SymbolSpace.RELATIONS

    def test_enum_iteration(self) -> None:
        """Test that we can iterate over all spaces."""
        spaces = list(SymbolSpace)
        assert len(spaces) == 6
        assert SymbolSpace.ENTITIES in spaces
        assert SymbolSpace.RELATIONS in spaces

    def test_value_reconstruction(self) -> None:
        """Test that we can reconstruct enum from value."""
        space = SymbolSpace("E")
        assert space == SymbolSpace.ENTITIES

        space = SymbolSpace("R")
        assert space == SymbolSpace.RELATIONS
