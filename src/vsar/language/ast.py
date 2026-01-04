"""AST node classes for VSARL."""

from typing import Any
from pydantic import BaseModel, Field


class Directive(BaseModel):
    """Configuration directive: @model FHRR(dim=8192, seed=1);"""

    name: str = Field(..., description="Directive name (e.g., 'model', 'threshold')")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters as key-value pairs"
    )

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"@{self.name}({params_str});"


class Fact(BaseModel):
    """Ground fact: fact parent(alice, bob) or fact ~enemy(alice, bob)."""

    predicate: str = Field(..., description="Predicate name (lowercase)")
    args: list[str] = Field(..., description="Arguments (all ground terms)")
    negated: bool = Field(default=False, description="True if this is a negative fact (~)")

    def __repr__(self) -> str:
        args_str = ", ".join(self.args)
        neg_str = "~" if self.negated else ""
        return f"fact {neg_str}{self.predicate}({args_str})."


class Query(BaseModel):
    """Query with variables: query parent(alice, X)? or query ~enemy(alice, X)?"""

    predicate: str = Field(..., description="Predicate name (lowercase)")
    args: list[str | None] = Field(..., description="Arguments (None = variable, str = constant)")
    negated: bool = Field(default=False, description="True if this is a negative query (~)")

    def __repr__(self) -> str:
        args_str = ", ".join(str(arg) if arg is not None else "?" for arg in self.args)
        neg_str = "~" if self.negated else ""
        return f"query {neg_str}{self.predicate}({args_str})?"

    def get_variables(self) -> list[int]:
        """Return positions of variables (0-indexed)."""
        return [i for i, arg in enumerate(self.args) if arg is None]

    def get_bound_args(self) -> dict[str, str]:
        """Return bound arguments as {position: value}."""
        return {str(i + 1): arg for i, arg in enumerate(self.args) if arg is not None}


class Atom(BaseModel):
    """Atom with variables: parent(X, bob) or parent(alice, Y)."""

    predicate: str = Field(..., description="Predicate name (lowercase)")
    args: list[str] = Field(..., description="Arguments (constants or variables)")

    def __repr__(self) -> str:
        args_str = ", ".join(self.args)
        return f"{self.predicate}({args_str})"

    def get_variables(self) -> list[str]:
        """Return list of variable names (uppercase args)."""
        return [arg for arg in self.args if arg[0].isupper()]

    def is_ground(self) -> bool:
        """Check if atom is ground (no variables)."""
        return len(self.get_variables()) == 0


class NAFLiteral(BaseModel):
    """Negation-as-failure literal: not enemy(X, _)"""

    atom: Atom = Field(..., description="Atom to negate")

    def __repr__(self) -> str:
        return f"not {repr(self.atom)}"

    def get_variables(self) -> list[str]:
        """Return list of variable names."""
        return self.atom.get_variables()


class Rule(BaseModel):
    """Horn rule: head :- body1, body2, not body3, ..."""

    head: Atom = Field(..., description="Head atom")
    body: list[Atom | NAFLiteral] = Field(..., description="Body atoms and NAF literals")

    def __repr__(self) -> str:
        body_str = ", ".join(repr(lit) for lit in self.body)
        return f"rule {repr(self.head)} :- {body_str}."


class Program(BaseModel):
    """Complete VSARL program."""

    directives: list[Directive] = Field(default_factory=list, description="Directives")
    facts: list[Fact] = Field(default_factory=list, description="Facts")
    rules: list[Rule] = Field(default_factory=list, description="Rules")
    queries: list[Query] = Field(default_factory=list, description="Queries")

    def __repr__(self) -> str:
        parts = []
        parts.extend(repr(d) for d in self.directives)
        parts.extend(repr(f) for f in self.facts)
        parts.extend(repr(r) for r in self.rules)
        parts.extend(repr(q) for q in self.queries)
        return "\n".join(parts)
