"""Tests for VSARL parser."""

import pytest
from vsar.language.ast import Directive, Fact, Program, Query
from vsar.language.parser import Parser, parse


class TestParser:
    """Test VSARL parser."""

    @pytest.fixture
    def parser(self) -> Parser:
        """Create parser instance."""
        return Parser()

    def test_parse_empty_program(self, parser: Parser) -> None:
        """Test parsing empty program."""
        program = parser.parse("")
        assert len(program.directives) == 0
        assert len(program.facts) == 0
        assert len(program.queries) == 0

    def test_parse_single_directive(self, parser: Parser) -> None:
        """Test parsing single directive."""
        text = "@model FHRR(dim=8192, seed=1);"
        program = parser.parse(text)
        assert len(program.directives) == 1
        directive = program.directives[0]
        assert directive.name == "model"
        assert directive.params == {"type": "FHRR", "dim": 8192, "seed": 1}

    def test_parse_multiple_directives(self, parser: Parser) -> None:
        """Test parsing multiple directives."""
        text = """
        @model FHRR(dim=8192, seed=1);
        @threshold(value=0.22);
        @beam(width=50);
        """
        program = parser.parse(text)
        assert len(program.directives) == 3
        assert program.directives[0].name == "model"
        assert program.directives[1].name == "threshold"
        assert program.directives[1].params == {"value": 0.22}
        assert program.directives[2].name == "beam"
        assert program.directives[2].params == {"width": 50}

    def test_parse_single_fact(self, parser: Parser) -> None:
        """Test parsing single fact."""
        text = "fact parent(alice, bob)."
        program = parser.parse(text)
        assert len(program.facts) == 1
        fact = program.facts[0]
        assert fact.predicate == "parent"
        assert fact.args == ["alice", "bob"]

    def test_parse_multiple_facts(self, parser: Parser) -> None:
        """Test parsing multiple facts."""
        text = """
        fact parent(alice, bob).
        fact parent(bob, carol).
        fact lives_in(alice, boston).
        """
        program = parser.parse(text)
        assert len(program.facts) == 3
        assert program.facts[0].predicate == "parent"
        assert program.facts[1].args == ["bob", "carol"]
        assert program.facts[2].predicate == "lives_in"

    def test_parse_unary_fact(self, parser: Parser) -> None:
        """Test parsing unary fact."""
        text = "fact person(alice)."
        program = parser.parse(text)
        assert len(program.facts) == 1
        assert program.facts[0].predicate == "person"
        assert program.facts[0].args == ["alice"]

    def test_parse_ternary_fact(self, parser: Parser) -> None:
        """Test parsing ternary fact."""
        text = "fact transfer(alice, bob, money)."
        program = parser.parse(text)
        assert len(program.facts) == 1
        assert program.facts[0].predicate == "transfer"
        assert program.facts[0].args == ["alice", "bob", "money"]

    def test_parse_query_with_variable(self, parser: Parser) -> None:
        """Test parsing query with variable."""
        text = "query parent(alice, X)?"
        program = parser.parse(text)
        assert len(program.queries) == 1
        query = program.queries[0]
        assert query.predicate == "parent"
        assert query.args == ["alice", None]
        assert query.get_variables() == [1]
        assert query.get_bound_args() == {"1": "alice"}

    def test_parse_query_with_multiple_variables(self, parser: Parser) -> None:
        """Test parsing query with multiple variables."""
        text = "query parent(X, Y)?"
        program = parser.parse(text)
        assert len(program.queries) == 1
        query = program.queries[0]
        assert query.args == [None, None]
        assert query.get_variables() == [0, 1]
        assert query.get_bound_args() == {}

    def test_parse_query_mixed_args(self, parser: Parser) -> None:
        """Test parsing query with mixed constants and variables."""
        text = "query transfer(alice, X, money)?"
        program = parser.parse(text)
        assert len(program.queries) == 1
        query = program.queries[0]
        assert query.args == ["alice", None, "money"]
        assert query.get_variables() == [1]
        assert query.get_bound_args() == {"1": "alice", "3": "money"}

    def test_parse_complete_program(self, parser: Parser) -> None:
        """Test parsing complete program with all elements."""
        text = """
        @model FHRR(dim=8192, seed=42);
        @threshold(value=0.22);

        fact parent(alice, bob).
        fact parent(bob, carol).

        query parent(alice, X)?
        query parent(X, carol)?
        """
        program = parser.parse(text)
        assert len(program.directives) == 2
        assert len(program.facts) == 2
        assert len(program.queries) == 2

    def test_parse_single_line_comment(self, parser: Parser) -> None:
        """Test parsing with single-line comments."""
        text = """
        // This is a comment
        fact parent(alice, bob). // Another comment
        """
        program = parser.parse(text)
        assert len(program.facts) == 1

    def test_parse_multi_line_comment(self, parser: Parser) -> None:
        """Test parsing with multi-line comments."""
        text = """
        /* This is a
           multi-line comment */
        fact parent(alice, bob).
        """
        program = parser.parse(text)
        assert len(program.facts) == 1

    def test_parse_string_value(self, parser: Parser) -> None:
        """Test parsing directive with string value."""
        text = '@name(value="test");'
        program = parser.parse(text)
        assert program.directives[0].params == {"value": "test"}

    def test_parse_boolean_values(self, parser: Parser) -> None:
        """Test parsing directive with boolean values."""
        text = "@config(enabled=true, disabled=false);"
        program = parser.parse(text)
        assert program.directives[0].params == {"enabled": True, "disabled": False}

    def test_parse_negative_number(self, parser: Parser) -> None:
        """Test parsing directive with negative number."""
        text = "@threshold(value=-0.5);"
        program = parser.parse(text)
        assert program.directives[0].params == {"value": -0.5}

    def test_parse_invalid_syntax(self, parser: Parser) -> None:
        """Test parsing invalid syntax raises error."""
        with pytest.raises(ValueError, match="Parse error"):
            parser.parse("invalid syntax here")

    def test_parse_fact_with_variable_raises_error(self, parser: Parser) -> None:
        """Test that facts with variables raise error."""
        text = "fact parent(alice, X)."
        with pytest.raises(ValueError, match="cannot contain variables"):
            parser.parse(text)

    def test_convenience_function(self) -> None:
        """Test convenience parse function."""
        text = "fact parent(alice, bob)."
        program = parse(text)
        assert len(program.facts) == 1
        assert program.facts[0].predicate == "parent"
