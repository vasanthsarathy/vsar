"""Program execution runner for VSAR IDE."""

from typing import Callable

from vsar.language.parser import parse
from vsar.semantics.engine import VSAREngine


class ProgramRunner:
    """Handles program execution."""

    def __init__(self, write_console: Callable[[str, str], None]):
        """Initialize program runner.

        Args:
            write_console: Callback function to write to console (text, tag)
        """
        self.write_console = write_console
        self.engine = None
        self.program = None

    def run(self, source_code: str) -> bool:
        """Run a VSAR program.

        Args:
            source_code: Source code to execute

        Returns:
            True if execution succeeded, False otherwise
        """
        try:
            # Parse program
            self.write_console("Parsing program...", "info")
            self.program = parse(source_code)

            # Create engine
            self.write_console("Initializing engine...", "info")
            self.engine = VSAREngine(self.program.directives)

            # Insert facts
            if self.program.facts:
                self.write_console(f"Inserting {len(self.program.facts)} facts...", "info")
                for fact in self.program.facts:
                    self.engine.insert_fact(fact)
                self.write_console(f"[OK] Inserted {len(self.program.facts)} facts", "success")

            # Show rules info
            if self.program.rules:
                self.write_console(f"Found {len(self.program.rules)} rules", "info")

            # Execute queries
            if not self.program.queries:
                self.write_console("No queries to execute", "warning")
                return True

            self.write_console(f"\nExecuting {len(self.program.queries)} queries...\n", "info")

            for i, query in enumerate(self.program.queries, 1):
                self._execute_query(i, query)

            self.write_console("\n[OK] Program execution completed", "success")
            return True

        except Exception as e:
            self.write_console(f"\n[ERROR] {e}", "error")
            return False

    def _execute_query(self, query_num: int, query) -> None:
        """Execute a single query.

        Args:
            query_num: Query number (for display)
            query: Query object to execute
        """
        try:
            # Format query
            args_str = ", ".join(arg if arg is not None else "X" for arg in query.args)
            query_str = f"{query.predicate}({args_str})"

            self.write_console(f"Query {query_num}: {query_str}", "info")

            # Execute query with rules if available
            result = self.engine.query(
                query, rules=self.program.rules if self.program.rules else None, k=10
            )

            # Display results
            if not result.results:
                self.write_console("  No results found", "warning")
            else:
                self.write_console(f"  Found {len(result.results)} results:", None)
                for entity, score in result.results:
                    # Color code scores
                    if score >= 0.9:
                        tag = "success"
                    elif score >= 0.7:
                        tag = None
                    else:
                        tag = "warning"

                    self.write_console(f"    {entity:<30} (score: {score:.4f})", tag)

            self.write_console("")  # Blank line

        except Exception as e:
            self.write_console(f"  [ERROR] Query failed: {e}", "error")

    def execute_single_query(self, query_str: str) -> bool:
        """Execute a single query against the current KB.

        Args:
            query_str: Query string (e.g., "parent(alice, X)?")

        Returns:
            True if execution succeeded, False otherwise
        """
        if not self.engine:
            self.write_console("No program loaded. Run a program first.", "error")
            return False

        try:
            # Parse query by wrapping in minimal program
            program_text = f"@model FHRR(dim=8192, seed=42);\nquery {query_str}"
            parsed_program = parse(program_text)

            if not parsed_program.queries:
                self.write_console("Invalid query syntax", "error")
                return False

            query = parsed_program.queries[0]

            # Execute query
            self.write_console(f"\nExecuting query: {query_str}\n", "info")
            self._execute_query(1, query)

            return True

        except Exception as e:
            self.write_console(f"[ERROR] Query failed: {e}", "error")
            return False

    def get_stats(self) -> dict:
        """Get KB statistics.

        Returns:
            Statistics dictionary or None if no engine loaded
        """
        if not self.engine:
            return None

        return self.engine.stats()
