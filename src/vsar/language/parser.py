"""VSARL parser using Lark."""

from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer, Tree
from lark.exceptions import UnexpectedInput, VisitError

from vsar.language.ast import Directive, Fact, Program, Query


class VSARLTransformer(Transformer):
    """Transform Lark parse tree to VSAR AST."""

    def start(self, items: list[Any]) -> Program:
        """Build Program from statements."""
        directives = []
        facts = []
        queries = []

        for item in items:
            if isinstance(item, Directive):
                directives.append(item)
            elif isinstance(item, Fact):
                facts.append(item)
            elif isinstance(item, Query):
                queries.append(item)

        return Program(directives=directives, facts=facts, queries=queries)

    def statement(self, items: list[Any]) -> Directive | Fact | Query:
        """Extract statement (directive, fact, or query)."""
        return items[0]

    def directive(self, items: list[Any]) -> Directive:
        """Build Directive from @name TYPE(params); or @name(params);"""
        name = str(items[0])

        # Check if there's a type parameter (e.g., @model FHRR(dim=8192))
        if len(items) > 1 and isinstance(items[1], str):
            # Format: @name TYPE(params)
            type_name = items[1]
            params = items[2] if len(items) > 2 else {}
            params["type"] = type_name
        else:
            # Format: @name(params)
            params = items[1] if len(items) > 1 else {}

        return Directive(name=name, params=params)

    def params(self, items: list[Any]) -> dict[str, Any]:
        """Build params dict from param list."""
        return dict(items)

    def param(self, items: list[Any]) -> tuple[str, Any]:
        """Build (key, value) tuple from param."""
        key = str(items[0])
        value = items[1]
        return (key, value)

    def value(self, items: list[Any]) -> Any:
        """Extract value from token."""
        token = items[0]
        if isinstance(token, Token):
            if token.type == "NUMBER":
                # Convert to int or float
                val_str = str(token)
                return int(val_str) if "." not in val_str else float(val_str)
            elif token.type == "STRING":
                # Remove quotes
                return str(token)[1:-1]
            elif token.type == "TRUE":
                return True
            elif token.type == "FALSE":
                return False
            else:
                # IDENTIFIER token
                return str(token)
        return token

    def fact(self, items: list[Any]) -> Fact:
        """Build Fact from fact atom."""
        predicate, args = items[0]
        # Ensure all args are strings (no None/variables in facts)
        if any(arg is None for arg in args):
            raise ValueError("Facts cannot contain variables")
        return Fact(predicate=predicate, args=args)

    def query(self, items: list[Any]) -> Query:
        """Build Query from query atom?"""
        predicate, args = items[0]
        return Query(predicate=predicate, args=args)

    def atom(self, items: list[Any]) -> tuple[str, list[str | None]]:
        """Build (predicate, args) from atom."""
        predicate = str(items[0])
        args = items[1] if len(items) > 1 else []
        return (predicate, args)

    def predicate(self, items: list[Any]) -> str:
        """Extract predicate name."""
        return str(items[0])

    def args(self, items: list[Any]) -> list[str | None]:
        """Build args list."""
        return items

    def arg(self, items: list[Any]) -> str | None:
        """Extract arg (constant or variable)."""
        return items[0]

    def constant(self, items: list[Any]) -> str:
        """Extract constant."""
        return str(items[0])

    def variable(self, items: list[Any]) -> None:
        """Extract variable (return None as placeholder)."""
        # Variables are represented as None in the Query args
        return None

    def LOWER_NAME(self, token: Token) -> str:
        """Extract lowercase name."""
        return str(token)

    def UPPER_NAME(self, token: Token) -> None:
        """Extract uppercase name (variable)."""
        # Return None to indicate this is a variable
        return None

    def IDENTIFIER(self, token: Token) -> str:
        """Extract identifier."""
        return str(token)


class Parser:
    """VSARL parser."""

    def __init__(self) -> None:
        """Initialize parser with grammar."""
        grammar_path = Path(__file__).parent / "grammar.lark"
        with open(grammar_path, "r", encoding="utf-8") as f:
            grammar = f.read()
        self.lark_parser = Lark(grammar, parser="lalr", start="start")
        self.transformer = VSARLTransformer()

    def parse(self, text: str) -> Program:
        """Parse VSARL text into Program AST.

        Args:
            text: VSARL source code

        Returns:
            Program AST

        Raises:
            ValueError: If parsing fails
        """
        try:
            tree = self.lark_parser.parse(text)
            program = self.transformer.transform(tree)
            return program
        except (UnexpectedInput, VisitError) as e:
            raise ValueError(f"Parse error: {e}") from e

    def parse_file(self, path: Path | str) -> Program:
        """Parse VSARL file into Program AST.

        Args:
            path: Path to .vsar file

        Returns:
            Program AST

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If parsing fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return self.parse(text)


# Convenience functions
_parser: Parser | None = None


def _get_parser() -> Parser:
    """Get singleton parser instance."""
    global _parser
    if _parser is None:
        _parser = Parser()
    return _parser


def parse(text: str) -> Program:
    """Parse VSARL text into Program AST.

    Args:
        text: VSARL source code

    Returns:
        Program AST
    """
    return _get_parser().parse(text)


def parse_file(path: Path | str) -> Program:
    """Parse VSARL file into Program AST.

    Args:
        path: Path to .vsar file

    Returns:
        Program AST
    """
    return _get_parser().parse_file(path)
