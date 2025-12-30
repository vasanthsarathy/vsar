"""Tests for VSAR loaders."""

from pathlib import Path

import pytest
from vsar.language.ast import Fact, Program
from vsar.language.loader import load_csv, load_facts, load_jsonl, load_vsar


class TestLoadCSV:
    """Test CSV loader."""

    def test_load_csv_with_predicate(self, tmp_path: Path) -> None:
        """Test loading CSV with predicate specified."""
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text("alice,bob\nbob,carol\n")

        facts = load_csv(csv_file, predicate="parent")
        assert len(facts) == 2
        assert facts[0].predicate == "parent"
        assert facts[0].args == ["alice", "bob"]
        assert facts[1].predicate == "parent"
        assert facts[1].args == ["bob", "carol"]

    def test_load_csv_without_predicate(self, tmp_path: Path) -> None:
        """Test loading CSV with predicate in first column."""
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text("parent,alice,bob\nparent,bob,carol\nlives_in,alice,boston\n")

        facts = load_csv(csv_file)
        assert len(facts) == 3
        assert facts[0].predicate == "parent"
        assert facts[0].args == ["alice", "bob"]
        assert facts[2].predicate == "lives_in"
        assert facts[2].args == ["alice", "boston"]

    def test_load_csv_with_spaces(self, tmp_path: Path) -> None:
        """Test CSV loading strips whitespace."""
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text(" alice , bob \n")

        facts = load_csv(csv_file, predicate="parent")
        assert facts[0].args == ["alice", "bob"]

    def test_load_csv_skip_empty_rows(self, tmp_path: Path) -> None:
        """Test CSV loading skips empty rows."""
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text("alice,bob\n\nbob,carol\n")

        facts = load_csv(csv_file, predicate="parent")
        assert len(facts) == 2

    def test_load_csv_unary_fact(self, tmp_path: Path) -> None:
        """Test loading unary facts."""
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text("alice\nbob\n")

        facts = load_csv(csv_file, predicate="person")
        assert len(facts) == 2
        assert facts[0].args == ["alice"]

    def test_load_csv_ternary_fact(self, tmp_path: Path) -> None:
        """Test loading ternary facts."""
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text("alice,bob,money\n")

        facts = load_csv(csv_file, predicate="transfer")
        assert len(facts) == 1
        assert facts[0].args == ["alice", "bob", "money"]

    def test_load_csv_file_not_found(self) -> None:
        """Test CSV loading raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent.csv")

    def test_load_csv_malformed_without_predicate(self, tmp_path: Path) -> None:
        """Test CSV loading raises error for malformed data."""
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text("parent\n")  # Only one column

        with pytest.raises(ValueError, match="at least 2 columns"):
            load_csv(csv_file)


class TestLoadJSONL:
    """Test JSONL loader."""

    def test_load_jsonl_basic(self, tmp_path: Path) -> None:
        """Test loading basic JSONL file."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text(
            '{"predicate": "parent", "args": ["alice", "bob"]}\n'
            '{"predicate": "parent", "args": ["bob", "carol"]}\n'
        )

        facts = load_jsonl(jsonl_file)
        assert len(facts) == 2
        assert facts[0].predicate == "parent"
        assert facts[0].args == ["alice", "bob"]
        assert facts[1].args == ["bob", "carol"]

    def test_load_jsonl_mixed_predicates(self, tmp_path: Path) -> None:
        """Test loading JSONL with different predicates."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text(
            '{"predicate": "parent", "args": ["alice", "bob"]}\n'
            '{"predicate": "lives_in", "args": ["alice", "boston"]}\n'
        )

        facts = load_jsonl(jsonl_file)
        assert len(facts) == 2
        assert facts[0].predicate == "parent"
        assert facts[1].predicate == "lives_in"

    def test_load_jsonl_skip_empty_lines(self, tmp_path: Path) -> None:
        """Test JSONL loading skips empty lines."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text(
            '{"predicate": "parent", "args": ["alice", "bob"]}\n'
            "\n"
            '{"predicate": "parent", "args": ["bob", "carol"]}\n'
        )

        facts = load_jsonl(jsonl_file)
        assert len(facts) == 2

    def test_load_jsonl_unary_fact(self, tmp_path: Path) -> None:
        """Test loading unary facts from JSONL."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text('{"predicate": "person", "args": ["alice"]}\n')

        facts = load_jsonl(jsonl_file)
        assert len(facts) == 1
        assert facts[0].args == ["alice"]

    def test_load_jsonl_ternary_fact(self, tmp_path: Path) -> None:
        """Test loading ternary facts from JSONL."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text('{"predicate": "transfer", "args": ["alice", "bob", "money"]}\n')

        facts = load_jsonl(jsonl_file)
        assert len(facts) == 1
        assert facts[0].args == ["alice", "bob", "money"]

    def test_load_jsonl_file_not_found(self) -> None:
        """Test JSONL loading raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_jsonl("nonexistent.jsonl")

    def test_load_jsonl_invalid_json(self, tmp_path: Path) -> None:
        """Test JSONL loading raises error for invalid JSON."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text("not json\n")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_jsonl(jsonl_file)

    def test_load_jsonl_not_object(self, tmp_path: Path) -> None:
        """Test JSONL loading raises error for non-object JSON."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text('["array", "not", "object"]\n')

        with pytest.raises(ValueError, match="Expected JSON object"):
            load_jsonl(jsonl_file)

    def test_load_jsonl_missing_predicate(self, tmp_path: Path) -> None:
        """Test JSONL loading raises error for missing predicate."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text('{"args": ["alice", "bob"]}\n')

        with pytest.raises(ValueError, match="Missing 'predicate'"):
            load_jsonl(jsonl_file)

    def test_load_jsonl_missing_args(self, tmp_path: Path) -> None:
        """Test JSONL loading raises error for missing args."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text('{"predicate": "parent"}\n')

        with pytest.raises(ValueError, match="Missing 'args'"):
            load_jsonl(jsonl_file)

    def test_load_jsonl_args_not_list(self, tmp_path: Path) -> None:
        """Test JSONL loading raises error for non-list args."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text('{"predicate": "parent", "args": "not a list"}\n')

        with pytest.raises(ValueError, match="'args' must be a list"):
            load_jsonl(jsonl_file)


class TestLoadVSAR:
    """Test VSAR loader."""

    def test_load_vsar_basic(self, tmp_path: Path) -> None:
        """Test loading VSAR file."""
        vsar_file = tmp_path / "program.vsar"
        vsar_file.write_text(
            "@model FHRR(dim=8192, seed=42);\n"
            "fact parent(alice, bob).\n"
            "query parent(alice, X)?\n"
        )

        program = load_vsar(vsar_file)
        assert isinstance(program, Program)
        assert len(program.directives) == 1
        assert len(program.facts) == 1
        assert len(program.queries) == 1

    def test_load_vsar_file_not_found(self) -> None:
        """Test VSAR loading raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_vsar("nonexistent.vsar")


class TestLoadFacts:
    """Test unified load_facts function."""

    def test_load_facts_auto_csv(self, tmp_path: Path) -> None:
        """Test auto-detection of CSV format."""
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text("parent,alice,bob\n")

        facts = load_facts(csv_file, format="auto")
        assert len(facts) == 1
        assert facts[0].predicate == "parent"

    def test_load_facts_auto_jsonl(self, tmp_path: Path) -> None:
        """Test auto-detection of JSONL format."""
        jsonl_file = tmp_path / "facts.jsonl"
        jsonl_file.write_text('{"predicate": "parent", "args": ["alice", "bob"]}\n')

        facts = load_facts(jsonl_file, format="auto")
        assert len(facts) == 1
        assert facts[0].predicate == "parent"

    def test_load_facts_auto_vsar(self, tmp_path: Path) -> None:
        """Test auto-detection of VSAR format."""
        vsar_file = tmp_path / "program.vsar"
        vsar_file.write_text("fact parent(alice, bob).\n")

        facts = load_facts(vsar_file, format="auto")
        assert len(facts) == 1
        assert facts[0].predicate == "parent"

    def test_load_facts_explicit_format(self, tmp_path: Path) -> None:
        """Test explicit format specification."""
        csv_file = tmp_path / "data.txt"  # Wrong extension
        csv_file.write_text("parent,alice,bob\n")

        facts = load_facts(csv_file, format="csv")
        assert len(facts) == 1

    def test_load_facts_csv_with_predicate(self, tmp_path: Path) -> None:
        """Test CSV loading with predicate parameter."""
        csv_file = tmp_path / "facts.csv"
        csv_file.write_text("alice,bob\n")

        facts = load_facts(csv_file, format="csv", predicate="parent")
        assert len(facts) == 1
        assert facts[0].predicate == "parent"

    def test_load_facts_auto_unknown_extension(self, tmp_path: Path) -> None:
        """Test auto-detection raises error for unknown extension."""
        file = tmp_path / "data.txt"
        file.write_text("data")

        with pytest.raises(ValueError, match="Cannot auto-detect format"):
            load_facts(file, format="auto")

    def test_load_facts_invalid_format(self, tmp_path: Path) -> None:
        """Test explicit invalid format raises error."""
        file = tmp_path / "data.txt"
        file.write_text("data")

        with pytest.raises(ValueError, match="Invalid format"):
            load_facts(file, format="invalid")
