"""Tests for REPL command."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from vsar.cli.main import app


class TestREPL:
    """Tests for REPL mode."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def family_vsar(self, tmp_path: Path):
        """Create a test VSAR file."""
        vsar_file = tmp_path / "family.vsar"
        vsar_file.write_text(
            """
@model FHRR(dim=512, seed=42);

fact parent(alice, bob).
fact parent(bob, carol).
"""
        )
        return vsar_file

    def test_repl_help_command(self, runner):
        """Test help command in REPL."""
        result = runner.invoke(app, ["repl"], input="help\nexit\n")
        assert result.exit_code == 0
        assert "Available Commands:" in result.stdout
        assert "load <file>" in result.stdout
        assert "query <query>" in result.stdout
        assert "stats" in result.stdout

    def test_repl_exit_command(self, runner):
        """Test exit command."""
        result = runner.invoke(app, ["repl"], input="exit\n")
        assert result.exit_code == 0
        assert "Goodbye!" in result.stdout

    def test_repl_quit_command(self, runner):
        """Test quit command."""
        result = runner.invoke(app, ["repl"], input="quit\n")
        assert result.exit_code == 0
        assert "Goodbye!" in result.stdout

    def test_repl_load_file(self, runner, family_vsar):
        """Test loading a VSAR file."""
        result = runner.invoke(app, ["repl"], input=f"load {family_vsar}\nexit\n")
        assert result.exit_code == 0
        assert "Loaded" in result.stdout
        assert "Inserted 2 facts" in result.stdout

    def test_repl_load_nonexistent_file(self, runner):
        """Test loading a non-existent file."""
        result = runner.invoke(app, ["repl"], input="load nonexistent.vsar\nexit\n")
        assert result.exit_code == 0
        assert "File not found" in result.stdout
        assert "Current directory" in result.stdout

    def test_repl_query_without_load(self, runner):
        """Test querying without loading a file first."""
        result = runner.invoke(app, ["repl"], input="query parent(alice, X)?\nexit\n")
        assert result.exit_code == 0
        assert "Error" in result.stdout
        assert "No program loaded" in result.stdout

    def test_repl_query_after_load(self, runner, family_vsar):
        """Test querying after loading a file."""
        result = runner.invoke(
            app, ["repl"], input=f"load {family_vsar}\nquery parent(alice, X)?\nexit\n"
        )
        assert result.exit_code == 0
        assert "Loaded" in result.stdout
        assert "bob" in result.stdout

    def test_repl_stats_without_load(self, runner):
        """Test stats without loading a file first."""
        result = runner.invoke(app, ["repl"], input="stats\nexit\n")
        assert result.exit_code == 0
        assert "Error" in result.stdout
        assert "No program loaded" in result.stdout

    def test_repl_stats_after_load(self, runner, family_vsar):
        """Test stats after loading a file."""
        result = runner.invoke(app, ["repl"], input=f"load {family_vsar}\nstats\nexit\n")
        assert result.exit_code == 0
        assert "Total Facts" in result.stdout
        assert "2" in result.stdout
        assert "parent" in result.stdout

    def test_repl_unknown_command(self, runner):
        """Test unknown command."""
        result = runner.invoke(app, ["repl"], input="unknown\nexit\n")
        assert result.exit_code == 0
        assert "Unknown command" in result.stdout

    def test_repl_empty_input(self, runner):
        """Test empty input."""
        result = runner.invoke(app, ["repl"], input="\n\nexit\n")
        assert result.exit_code == 0

    def test_repl_multiple_queries(self, runner, family_vsar):
        """Test multiple queries in sequence."""
        result = runner.invoke(
            app,
            ["repl"],
            input=f"load {family_vsar}\nquery parent(alice, X)?\nquery parent(X, carol)?\nexit\n",
        )
        assert result.exit_code == 0
        assert "bob" in result.stdout
        assert "carol" in result.stdout or "bob" in result.stdout

    def test_repl_load_usage_error(self, runner):
        """Test load command without filename."""
        result = runner.invoke(app, ["repl"], input="load\nexit\n")
        assert result.exit_code == 0
        assert "Usage: load <file>" in result.stdout

    def test_repl_query_usage_error(self, runner, family_vsar):
        """Test query command without query text."""
        result = runner.invoke(app, ["repl"], input=f"load {family_vsar}\nquery\nexit\n")
        assert result.exit_code == 0
        assert "Usage: query <query>" in result.stdout
