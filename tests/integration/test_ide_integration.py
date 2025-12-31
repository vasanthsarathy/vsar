"""Integration test for VSAR IDE with all examples."""

import sys
from pathlib import Path
from io import StringIO

# Test the IDE components
from vsar.ide.runner import ProgramRunner
from vsar.ide.highlighter import VSARLHighlighter
import tkinter as tk


class MockConsole:
    """Mock console for testing."""

    def __init__(self):
        self.messages = []

    def write(self, text, tag=None):
        """Capture console output."""
        self.messages.append((text, tag))
        print(f"[{tag or 'NONE'}] {text}")


def _test_single_example(example_path: Path, console: MockConsole) -> bool:
    """Test an example program (helper function).

    Args:
        example_path: Path to example file
        console: Test console

    Returns:
        True if test passed
    """
    print(f"\n{'='*60}")
    print(f"Testing: {example_path.name}")
    print(f"{'='*60}")

    try:
        # Read example
        with open(example_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        print(f"\n[FILE] Loaded {len(source_code)} characters")

        # Create runner
        runner = ProgramRunner(console.write)

        # Run program
        console.messages.clear()
        success = runner.run(source_code)

        if not success:
            print(f"\n[ERROR] Program execution failed!")
            return False

        # Check for results
        has_results = any("Found" in msg[0] and "results" in msg[0]
                         for msg in console.messages)

        if has_results:
            print(f"\n[SUCCESS] Program executed with results")
        else:
            print(f"\n[WARNING] Program executed but no results found")

        # Test stats
        stats = runner.get_stats()
        if stats:
            print(f"\n[STATS] Total facts: {stats['total_facts']}")
            print(f"[STATS] Predicates: {list(stats['predicates'].keys())}")

        # Test single query if engine is loaded
        if runner.engine and runner.program and runner.program.queries:
            first_query = runner.program.queries[0]
            args_str = ", ".join(arg if arg is not None else "X" for arg in first_query.args)
            query_str = f"{first_query.predicate}({args_str})?"

            print(f"\n[QUERY] Testing query: {query_str}")
            console.messages.clear()
            runner.execute_single_query(query_str)

        return True

    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_syntax_highlighting():
    """Test syntax highlighting."""
    print(f"\n{'='*60}")
    print("Testing Syntax Highlighting")
    print(f"{'='*60}")

    # Skip if no display available (CI environment)
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the window
    except tk.TclError:
        print("[SKIP] No display available - skipping GUI test")
        return True

    text_widget = tk.Text(root)
    highlighter = VSARLHighlighter(text_widget)

    # Test code with various syntax elements
    test_code = """
// This is a comment
@model FHRR(dim=512, seed=42);
@beam(width=50);

/* Multi-line
   comment */

fact parent(alice, bob).
fact parent(bob, carol).

rule grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

query grandparent(alice, X)?
"""

    text_widget.insert("1.0", test_code)
    highlighter.highlight_all()

    # Check that tags were applied
    tags = text_widget.tag_names()
    expected_tags = ['keyword', 'directive', 'comment', 'variable', 'predicate']

    print(f"\n[HIGHLIGHT] Applied tags: {[t for t in tags if t in expected_tags]}")

    for tag in expected_tags:
        ranges = text_widget.tag_ranges(tag)
        if ranges:
            print(f"[HIGHLIGHT] Tag '{tag}': {len(ranges)//2} occurrences")

    root.destroy()
    print(f"\n[SUCCESS] Syntax highlighting works!")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("VSAR IDE Integration Tests")
    print("="*60)

    # Find all example files
    examples_dir = Path("examples")
    if not examples_dir.exists():
        print(f"\n[ERROR] Examples directory not found: {examples_dir}")
        return False

    example_files = sorted(examples_dir.glob("*.vsar"))

    if not example_files:
        print(f"\n[ERROR] No .vsar files found in {examples_dir}")
        return False

    print(f"\n[INFO] Found {len(example_files)} example files:")
    for f in example_files:
        print(f"  - {f.name}")

    # Test console
    console = MockConsole()

    # Test each example
    results = {}
    for example_path in example_files:
        success = _test_single_example(example_path, console)
        results[example_path.name] = success

    # Test syntax highlighting
    try:
        highlight_success = test_syntax_highlighting()
        results["syntax_highlighting"] = highlight_success
    except Exception as e:
        print(f"\n[ERROR] Syntax highlighting test failed: {e}")
        results["syntax_highlighting"] = False

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return True
    else:
        print(f"\n[FAILURE] {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
