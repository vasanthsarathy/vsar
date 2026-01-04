"""Syntax highlighter for VSARL."""

import re
import tkinter as tk
from typing import Pattern


class VSARLHighlighter:
    """Syntax highlighter for VSARL code."""

    def __init__(self, text_widget: tk.Text):
        """Initialize highlighter.

        Args:
            text_widget: Tkinter Text widget to highlight
        """
        self.text = text_widget
        self._setup_tags()
        self._compile_patterns()

    def _setup_tags(self) -> None:
        """Set up text tags for different syntax elements."""
        # Keywords
        self.text.tag_config("keyword", foreground="#0000FF")  # Blue

        # Directives
        self.text.tag_config("directive", foreground="#9400D3")  # Purple

        # Comments
        self.text.tag_config("comment", foreground="#808080")  # Gray

        # Strings
        self.text.tag_config("string", foreground="#008000")  # Green

        # Variables (uppercase identifiers)
        self.text.tag_config("variable", foreground="#FF8C00")  # Orange

        # Predicates (lowercase identifiers)
        self.text.tag_config("predicate", foreground="#000000")  # Black

        # Negation operators (~, not)
        self.text.tag_config("negation", foreground="#DC143C", font=("Courier", 10, "bold"))  # Crimson, bold

    def _compile_patterns(self) -> None:
        """Compile regex patterns for syntax elements."""
        # Keywords (including 'not' for NAF)
        self.keyword_pattern = re.compile(r"\b(fact|rule|query|not)\b")

        # Negation prefix (~) before predicates
        self.negation_prefix_pattern = re.compile(r"~(?=[a-z])")

        # Directives (@ followed by word)
        self.directive_pattern = re.compile(r"@\w+")

        # Single-line comments
        self.comment_line_pattern = re.compile(r"//.*$", re.MULTILINE)

        # Multi-line comments
        self.comment_block_pattern = re.compile(r"/\*.*?\*/", re.DOTALL)

        # Strings (double quotes)
        self.string_pattern = re.compile(r'"[^"]*"')

        # Variables (uppercase start, e.g., X, Person, Variable)
        self.variable_pattern = re.compile(r"\b[A-Z][a-zA-Z0-9_]*\b")

        # Predicates (lowercase start, e.g., parent, lives_in)
        self.predicate_pattern = re.compile(r"\b[a-z][a-z0-9_]*\b")

    def highlight_all(self) -> None:
        """Apply syntax highlighting to entire text."""
        # Remove existing tags
        for tag in ["keyword", "directive", "comment", "string", "variable", "predicate", "negation"]:
            self.text.tag_remove(tag, "1.0", tk.END)

        # Get all text
        content = self.text.get("1.0", tk.END)

        # Apply highlighting in order (later tags override earlier ones)
        # Order matters: apply most specific patterns last

        # 1. Comments (highest priority - should override everything)
        self._apply_pattern(content, self.comment_block_pattern, "comment")
        self._apply_pattern(content, self.comment_line_pattern, "comment")

        # 2. Strings
        self._apply_pattern(content, self.string_pattern, "string")

        # 3. Directives
        self._apply_pattern(content, self.directive_pattern, "directive")

        # 4. Negation prefix (~)
        self._apply_pattern(content, self.negation_prefix_pattern, "negation")

        # 5. Keywords (including 'not')
        self._apply_pattern(content, self.keyword_pattern, "keyword")

        # 6. Variables
        self._apply_pattern(content, self.variable_pattern, "variable")

        # 7. Predicates (lowest priority)
        # Skip predicates that are part of keywords or directives
        self._apply_predicate_pattern(content)

    def _apply_pattern(self, content: str, pattern: Pattern, tag: str) -> None:
        """Apply a regex pattern and tag matches.

        Args:
            content: Text content to search
            pattern: Compiled regex pattern
            tag: Tag name to apply
        """
        for match in pattern.finditer(content):
            start_idx = f"1.0+{match.start()}c"
            end_idx = f"1.0+{match.end()}c"
            self.text.tag_add(tag, start_idx, end_idx)

    def _apply_predicate_pattern(self, content: str) -> None:
        """Apply predicate pattern, avoiding keywords.

        Args:
            content: Text content to search
        """
        keywords = {"fact", "rule", "query"}

        for match in self.predicate_pattern.finditer(content):
            word = match.group()
            # Skip if it's a keyword
            if word not in keywords:
                start_idx = f"1.0+{match.start()}c"
                end_idx = f"1.0+{match.end()}c"
                self.text.tag_add("predicate", start_idx, end_idx)

    def highlight_range(self, start: str, end: str) -> None:
        """Highlight a specific range of text.

        Args:
            start: Start index (e.g., "1.0")
            end: End index (e.g., "end")
        """
        # For now, just rehighlight everything
        # This could be optimized to only highlight the changed range
        self.highlight_all()
