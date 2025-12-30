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
    """Ground fact: fact parent(alice, bob)."""

    predicate: str = Field(..., description="Predicate name (lowercase)")
    args: list[str] = Field(..., description="Arguments (all ground terms)")

    def __repr__(self) -> str:
        args_str = ", ".join(self.args)
        return f"fact {self.predicate}({args_str})."


class Query(BaseModel):
    """Query with variables: query parent(alice, X)?"""

    predicate: str = Field(..., description="Predicate name (lowercase)")
    args: list[str | None] = Field(
        ..., description="Arguments (None = variable, str = constant)"
    )

    def __repr__(self) -> str:
        args_str = ", ".join(str(arg) if arg is not None else "?" for arg in self.args)
        return f"query {self.predicate}({args_str})?"

    def get_variables(self) -> list[int]:
        """Return positions of variables (0-indexed)."""
        return [i for i, arg in enumerate(self.args) if arg is None]

    def get_bound_args(self) -> dict[str, str]:
        """Return bound arguments as {position: value}."""
        return {str(i + 1): arg for i, arg in enumerate(self.args) if arg is not None}


class Program(BaseModel):
    """Complete VSARL program."""

    directives: list[Directive] = Field(default_factory=list, description="Directives")
    facts: list[Fact] = Field(default_factory=list, description="Facts")
    queries: list[Query] = Field(default_factory=list, description="Queries")

    def __repr__(self) -> str:
        parts = []
        parts.extend(repr(d) for d in self.directives)
        parts.extend(repr(f) for f in self.facts)
        parts.extend(repr(q) for q in self.queries)
        return "\n".join(parts)
