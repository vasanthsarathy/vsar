"""VSAR IDE main application."""

import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional

from vsar.ide.highlighter import VSARLHighlighter
from vsar.ide.runner import ProgramRunner


class VSARIDE:
    """Main IDE application window."""

    def __init__(self, root: tk.Tk):
        """Initialize the IDE.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("VSAR IDE")
        self.root.geometry("1200x800")

        # State
        self.current_file: Optional[Path] = None
        self.modified = False

        # Initialize program runner
        self.runner = ProgramRunner(self.write_console)

        # Build UI
        self._create_menu()
        self._create_toolbar()
        self._create_main_area()
        self._create_status_bar()

        # Bind events
        self.editor.bind("<<Modified>>", self._on_text_modified)
        self.editor.bind("<KeyRelease>", self._update_cursor_position)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Show welcome message
        self._show_welcome()

    def _create_menu(self) -> None:
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(
            label="Save As...", command=self.save_file_as, accelerator="Ctrl+Shift+S"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Cut", command=lambda: self.editor.event_generate("<<Cut>>"))
        edit_menu.add_command(
            label="Copy", command=lambda: self.editor.event_generate("<<Copy>>")
        )
        edit_menu.add_command(
            label="Paste", command=lambda: self.editor.event_generate("<<Paste>>")
        )
        edit_menu.add_separator()
        edit_menu.add_command(
            label="Select All", command=lambda: self.editor.event_generate("<<SelectAll>>")
        )

        # Run menu
        run_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Run", menu=run_menu)
        run_menu.add_command(label="Run Program", command=self.run_program, accelerator="F5")
        run_menu.add_command(label="Run Query...", command=self.run_query, accelerator="Ctrl+Q")
        run_menu.add_separator()
        run_menu.add_command(label="Show Stats", command=self.show_stats)
        run_menu.add_command(label="Clear Console", command=self.clear_console)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)

        # Bind keyboard shortcuts
        self.root.bind("<Control-n>", lambda e: self.new_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-Shift-S>", lambda e: self.save_file_as())
        self.root.bind("<F5>", lambda e: self.run_program())
        self.root.bind("<Control-q>", lambda e: self.run_query())

    def _create_toolbar(self) -> None:
        """Create toolbar."""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)

        # Buttons
        ttk.Button(toolbar, text="📄 New", command=self.new_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📁 Open", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Save", command=self.save_file).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(toolbar, text="▶️ Run", command=self.run_program).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔍 Query", command=self.run_query).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📊 Stats", command=self.show_stats).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(toolbar, text="🗑️ Clear", command=self.clear_console).pack(
            side=tk.LEFT, padx=2
        )

    def _create_main_area(self) -> None:
        """Create main split pane area."""
        # Create PanedWindow
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left pane - Editor
        editor_frame = ttk.Frame(paned)
        paned.add(editor_frame, weight=1)

        editor_label = ttk.Label(editor_frame, text="Editor", font=("Arial", 10, "bold"))
        editor_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        self.editor = scrolledtext.ScrolledText(
            editor_frame,
            wrap=tk.NONE,
            font=("Courier New", 11),
            undo=True,
            maxundo=-1,
        )
        self.editor.pack(fill=tk.BOTH, expand=True)

        # Right pane - Console
        console_frame = ttk.Frame(paned)
        paned.add(console_frame, weight=1)

        console_label = ttk.Label(console_frame, text="Console", font=("Arial", 10, "bold"))
        console_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        self.console = scrolledtext.ScrolledText(
            console_frame,
            wrap=tk.WORD,
            font=("Courier New", 10),
            state=tk.DISABLED,
            bg="#f0f0f0",
        )
        self.console.pack(fill=tk.BOTH, expand=True)

        # Configure console tags for colored output
        self.console.tag_config("error", foreground="red")
        self.console.tag_config("success", foreground="green")
        self.console.tag_config("info", foreground="blue")
        self.console.tag_config("warning", foreground="orange")

        # Initialize syntax highlighter
        self.highlighter = VSARLHighlighter(self.editor)
        self._highlight_after_id = None

    def _create_status_bar(self) -> None:
        """Create status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.file_label = ttk.Label(status_frame, text="File: untitled.vsar", relief=tk.SUNKEN)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.cursor_label = ttk.Label(status_frame, text="Line: 1  Col: 1", relief=tk.SUNKEN)
        self.cursor_label.pack(side=tk.LEFT, padx=2)

        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, padx=2)

    def _on_text_modified(self, event: tk.Event) -> None:
        """Handle text modification."""
        if self.editor.edit_modified():
            if not self.modified:
                self.modified = True
                self._update_title()
            self.editor.edit_modified(False)

            # Debounced syntax highlighting
            if self._highlight_after_id:
                self.root.after_cancel(self._highlight_after_id)
            self._highlight_after_id = self.root.after(300, self._apply_highlighting)

    def _apply_highlighting(self) -> None:
        """Apply syntax highlighting."""
        self.highlighter.highlight_all()
        self._highlight_after_id = None

    def _update_cursor_position(self, event: tk.Event) -> None:
        """Update cursor position in status bar."""
        cursor_pos = self.editor.index(tk.INSERT)
        line, col = cursor_pos.split(".")
        self.cursor_label.config(text=f"Line: {line}  Col: {int(col) + 1}")

    def _update_title(self) -> None:
        """Update window title."""
        filename = self.current_file.name if self.current_file else "untitled.vsar"
        modified_mark = "*" if self.modified else ""
        self.root.title(f"VSAR IDE - {filename}{modified_mark}")

    def _update_file_label(self) -> None:
        """Update file label in status bar."""
        filename = str(self.current_file) if self.current_file else "untitled.vsar"
        self.file_label.config(text=f"File: {filename}")

    def write_console(self, text: str, tag: Optional[str] = None) -> None:
        """Write text to console.

        Args:
            text: Text to write
            tag: Optional tag for styling (error, success, info, warning)
        """
        self.console.config(state=tk.NORMAL)
        if tag:
            self.console.insert(tk.END, text + "\n", tag)
        else:
            self.console.insert(tk.END, text + "\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def clear_console(self) -> None:
        """Clear console output."""
        self.console.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)
        self.console.config(state=tk.DISABLED)
        self.write_console("Console cleared", "info")

    def new_file(self) -> None:
        """Create a new file."""
        if self.modified:
            response = messagebox.askyesnocancel(
                "Save Changes", "Do you want to save changes to the current file?"
            )
            if response is None:  # Cancel
                return
            if response:  # Yes
                self.save_file()

        self.editor.delete(1.0, tk.END)
        self.current_file = None
        self.modified = False
        self._update_title()
        self._update_file_label()
        self.write_console("New file created", "info")

    def open_file(self) -> None:
        """Open a file."""
        filename = filedialog.askopenfilename(
            title="Open VSAR File",
            filetypes=[("VSAR Files", "*.vsar"), ("All Files", "*.*")],
        )
        if not filename:
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()

            self.editor.delete(1.0, tk.END)
            self.editor.insert(1.0, content)
            self.current_file = Path(filename)
            self.modified = False
            self.editor.edit_modified(False)
            self._update_title()
            self._update_file_label()
            self.write_console(f"Opened: {filename}", "success")

            # Apply syntax highlighting
            self.highlighter.highlight_all()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file:\n{e}")
            self.write_console(f"Error opening file: {e}", "error")

    def save_file(self) -> None:
        """Save the current file."""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_file_as()

    def save_file_as(self) -> None:
        """Save the current file with a new name."""
        filename = filedialog.asksaveasfilename(
            title="Save VSAR File",
            defaultextension=".vsar",
            filetypes=[("VSAR Files", "*.vsar"), ("All Files", "*.*")],
        )
        if not filename:
            return

        self._save_to_file(Path(filename))

    def _save_to_file(self, path: Path) -> None:
        """Save content to file.

        Args:
            path: Path to save to
        """
        try:
            content = self.editor.get(1.0, tk.END)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            self.current_file = path
            self.modified = False
            self.editor.edit_modified(False)
            self._update_title()
            self._update_file_label()
            self.write_console(f"Saved: {path}", "success")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
            self.write_console(f"Error saving file: {e}", "error")

    def run_program(self) -> None:
        """Run the current program."""
        self.write_console("=" * 60, "info")
        self.write_console("Running program...", "info")
        self.write_console("=" * 60, "info")

        # Get source code from editor
        source_code = self.editor.get("1.0", tk.END)

        # Run program
        self.status_label.config(text="Running...")
        self.root.update_idletasks()

        success = self.runner.run(source_code)

        if success:
            self.status_label.config(text="Ready")
        else:
            self.status_label.config(text="Error")

    def run_query(self) -> None:
        """Run a query."""
        # Show dialog for query input
        dialog = tk.Toplevel(self.root)
        dialog.title("Run Query")
        dialog.geometry("500x150")
        dialog.transient(self.root)
        dialog.grab_set()

        # Query input
        ttk.Label(dialog, text="Enter query (e.g., parent(alice, X)?):").pack(
            pady=(10, 5), padx=10, anchor=tk.W
        )

        query_entry = ttk.Entry(dialog, width=60)
        query_entry.pack(pady=5, padx=10)
        query_entry.focus()

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        def execute():
            query_str = query_entry.get().strip()
            if query_str:
                self.runner.execute_single_query(query_str)
            dialog.destroy()

        def cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="Execute", command=execute).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)

        # Bind Enter key to execute
        query_entry.bind("<Return>", lambda e: execute())
        dialog.bind("<Escape>", lambda e: cancel())

    def show_stats(self) -> None:
        """Show KB statistics."""
        stats = self.runner.get_stats()

        if not stats:
            self.write_console("No program loaded. Run a program first.", "warning")
            return

        self.write_console("\n" + "=" * 60, "info")
        self.write_console("Knowledge Base Statistics", "info")
        self.write_console("=" * 60, "info")
        self.write_console(f"Total Facts: {stats['total_facts']}", None)
        self.write_console(f"Predicates: {len(stats['predicates'])}", None)

        if stats["predicates"]:
            self.write_console("\nPredicate Details:", None)
            for predicate, count in sorted(stats["predicates"].items()):
                self.write_console(f"  {predicate:<20} {count} facts", None)

        self.write_console("=" * 60 + "\n", "info")

    def _show_welcome(self) -> None:
        """Show welcome message in console."""
        self.write_console("=" * 60, "info")
        self.write_console("Welcome to VSAR IDE", "success")
        self.write_console("=" * 60, "info")
        self.write_console("", None)
        self.write_console("VSA-grounded Reasoning Language - Interactive Development Environment", None)
        self.write_console("", None)
        self.write_console("Quick Start:", "info")
        self.write_console("  1. Create a new file (Ctrl+N) or open an example (Ctrl+O)", None)
        self.write_console("  2. Write your VSAR program with facts, rules, and queries", None)
        self.write_console("  3. Run the program (F5) to see results", None)
        self.write_console("  4. Execute queries interactively (Ctrl+Q)", None)
        self.write_console("", None)
        self.write_console("Need help? Press Help > Keyboard Shortcuts", "info")
        self.write_console("=" * 60 + "\n", "info")

    def show_shortcuts(self) -> None:
        """Show keyboard shortcuts help."""
        shortcuts = """
Keyboard Shortcuts:

File Operations:
  Ctrl+N - New file
  Ctrl+O - Open file
  Ctrl+S - Save file
  Ctrl+Shift+S - Save as

Run Operations:
  F5 - Run program
  Ctrl+Q - Run query

Console:
  Clear - Clear console output
"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def show_about(self) -> None:
        """Show about dialog."""
        about_text = """
VSAR IDE
Version 0.1.0

Interactive Development Environment for VSARL
(VSA-grounded Reasoning Language)

Built with Python and Tkinter
"""
        messagebox.showinfo("About VSAR IDE", about_text)

    def _on_closing(self) -> None:
        """Handle window close event."""
        if self.modified:
            response = messagebox.askyesnocancel(
                "Save Changes", "Do you want to save changes before closing?"
            )
            if response is None:  # Cancel
                return
            if response:  # Yes
                self.save_file()

        self.root.destroy()


def main() -> None:
    """Main entry point for VSAR IDE."""
    root = tk.Tk()
    app = VSARIDE(root)
    root.mainloop()


if __name__ == "__main__":
    main()
