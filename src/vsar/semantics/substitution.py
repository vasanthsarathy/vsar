"""Variable substitution for rule execution."""

from typing import Any

from pydantic import BaseModel, Field

from vsar.language.ast import Atom


class Substitution(BaseModel):
    """Variable substitution mapping variables to values.

    Example:
        sub = Substitution()
        sub.bind("X", "alice")
        sub.bind("Y", "bob")
        sub.get("X")  # Returns "alice"
    """

    bindings: dict[str, str] = Field(default_factory=dict, description="Variable → value mappings")

    def bind(self, var: str, value: str) -> "Substitution":
        """Add a variable binding.

        Args:
            var: Variable name (e.g., "X", "Y")
            value: Value to bind (e.g., "alice", "bob")

        Returns:
            New Substitution with the binding added
        """
        new_bindings = self.bindings.copy()
        new_bindings[var] = value
        return Substitution(bindings=new_bindings)

    def get(self, var: str) -> str | None:
        """Get the value bound to a variable.

        Args:
            var: Variable name

        Returns:
            Bound value, or None if variable is unbound
        """
        return self.bindings.get(var)

    def has(self, var: str) -> bool:
        """Check if a variable is bound.

        Args:
            var: Variable name

        Returns:
            True if variable is bound
        """
        return var in self.bindings

    def vars(self) -> list[str]:
        """Get all variables in the substitution.

        Returns:
            List of variable names
        """
        return list(self.bindings.keys())

    def is_empty(self) -> bool:
        """Check if substitution is empty.

        Returns:
            True if no bindings exist
        """
        return len(self.bindings) == 0

    def apply_to_atom(self, atom: Atom) -> Atom:
        """Apply substitution to an atom, replacing variables with their values.

        Args:
            atom: Atom to substitute into

        Returns:
            New atom with variables replaced

        Example:
            sub = Substitution(bindings={"X": "alice", "Y": "bob"})
            atom = Atom(predicate="parent", args=["X", "Y"])
            result = sub.apply_to_atom(atom)
            # result = Atom(predicate="parent", args=["alice", "bob"])
        """
        new_args = []
        for arg in atom.args:
            # If arg is a variable and is bound, replace it
            if arg[0].isupper() and arg in self.bindings:
                new_args.append(self.bindings[arg])
            else:
                # Otherwise keep the original arg
                new_args.append(arg)

        return Atom(predicate=atom.predicate, args=new_args)

    def compose(self, other: "Substitution") -> "Substitution":
        """Compose two substitutions.

        The result applies this substitution first, then the other.
        If both bind the same variable, other's binding takes precedence.

        Args:
            other: Substitution to compose with

        Returns:
            Composed substitution
        """
        # Start with this substitution's bindings
        new_bindings = self.bindings.copy()
        # Add other's bindings (overwriting conflicts)
        new_bindings.update(other.bindings)
        return Substitution(bindings=new_bindings)

    def __repr__(self) -> str:
        """String representation."""
        if self.is_empty():
            return "Substitution({})"
        items = ", ".join(f"{var}={val}" for var, val in sorted(self.bindings.items()))
        return f"Substitution({{{items}}})"


def is_variable(term: str) -> bool:
    """Check if a term is a variable (starts with uppercase).

    Args:
        term: Term to check

    Returns:
        True if term is a variable
    """
    return len(term) > 0 and term[0].isupper()


def get_atom_variables(atom: Atom) -> list[str]:
    """Get all variables in an atom.

    Args:
        atom: Atom to analyze

    Returns:
        List of variable names (may contain duplicates)
    """
    return [arg for arg in atom.args if is_variable(arg)]


def get_atom_unique_variables(atom: Atom) -> list[str]:
    """Get unique variables in an atom.

    Args:
        atom: Atom to analyze

    Returns:
        List of unique variable names
    """
    seen = set()
    result = []
    for arg in atom.args:
        if is_variable(arg) and arg not in seen:
            seen.add(arg)
            result.append(arg)
    return result
