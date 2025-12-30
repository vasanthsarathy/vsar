"""VSAR fact loaders for CSV, JSONL, and VSAR files."""

import csv
import json
from pathlib import Path
from typing import Any

from vsar.language.ast import Fact, Program
from vsar.language.parser import parse_file


def load_csv(path: Path | str, predicate: str | None = None) -> list[Fact]:
    """Load facts from CSV file.

    Format: Each row is a fact. If predicate is provided, all rows use that predicate.
    Otherwise, first column is the predicate name.

    Examples:
        With predicate="parent":
            alice,bob
            bob,carol

        Without predicate:
            parent,alice,bob
            parent,bob,carol
            lives_in,alice,boston

    Args:
        path: Path to CSV file
        predicate: Optional predicate name. If None, first column is predicate.

    Returns:
        List of Fact objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    facts = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for line_num, row in enumerate(reader, start=1):
                if not row:
                    continue  # Skip empty rows

                if predicate is not None:
                    # All args are in the row
                    args = [cell.strip() for cell in row]
                    facts.append(Fact(predicate=predicate, args=args))
                else:
                    # First column is predicate
                    if len(row) < 2:
                        raise ValueError(
                            f"Line {line_num}: Expected at least 2 columns (predicate + args)"
                        )
                    pred = row[0].strip()
                    args = [cell.strip() for cell in row[1:]]
                    facts.append(Fact(predicate=pred, args=args))

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Failed to parse CSV: {e}") from e

    return facts


def load_jsonl(path: Path | str) -> list[Fact]:
    """Load facts from JSONL file.

    Format: Each line is a JSON object with "predicate" and "args" fields.

    Example:
        {"predicate": "parent", "args": ["alice", "bob"]}
        {"predicate": "parent", "args": ["bob", "carol"]}
        {"predicate": "lives_in", "args": ["alice", "boston"]}

    Args:
        path: Path to JSONL file

    Returns:
        List of Fact objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSONL is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    facts = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {line_num}: Invalid JSON: {e}") from e

                # Validate structure
                if not isinstance(obj, dict):
                    raise ValueError(f"Line {line_num}: Expected JSON object, got {type(obj)}")

                if "predicate" not in obj:
                    raise ValueError(f"Line {line_num}: Missing 'predicate' field")

                if "args" not in obj:
                    raise ValueError(f"Line {line_num}: Missing 'args' field")

                if not isinstance(obj["args"], list):
                    raise ValueError(
                        f"Line {line_num}: 'args' must be a list, got {type(obj['args'])}"
                    )

                # Create fact
                predicate = str(obj["predicate"])
                args = [str(arg) for arg in obj["args"]]
                facts.append(Fact(predicate=predicate, args=args))

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Failed to parse JSONL: {e}") from e

    return facts


def load_vsar(path: Path | str) -> Program:
    """Load VSAR program from .vsar file.

    This is a convenience wrapper around parse_file.

    Args:
        path: Path to .vsar file

    Returns:
        Program AST

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If parsing fails
    """
    return parse_file(path)


def load_facts(
    path: Path | str, format: str = "auto", predicate: str | None = None
) -> list[Fact]:
    """Load facts from file with automatic format detection.

    Args:
        path: Path to file
        format: File format ("csv", "jsonl", "vsar", or "auto"). Auto detects from extension.
        predicate: Optional predicate name for CSV files

    Returns:
        List of Fact objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is invalid or file is malformed
    """
    path = Path(path)

    # Auto-detect format from extension
    if format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".csv":
            format = "csv"
        elif suffix in [".jsonl", ".json"]:
            format = "jsonl"
        elif suffix == ".vsar":
            format = "vsar"
        else:
            raise ValueError(
                f"Cannot auto-detect format from extension '{suffix}'. "
                "Specify format explicitly."
            )

    # Load facts based on format
    if format == "csv":
        return load_csv(path, predicate=predicate)
    elif format == "jsonl":
        return load_jsonl(path)
    elif format == "vsar":
        program = load_vsar(path)
        return program.facts
    else:
        raise ValueError(f"Invalid format: {format}. Expected 'csv', 'jsonl', or 'vsar'.")
