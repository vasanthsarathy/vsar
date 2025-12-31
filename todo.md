# TODO: VSAR IDE Implementation

## Problem
VSAR currently has a CLI interface for running programs and queries. Users would benefit from a dedicated IDE (similar to DrRacket) that provides:
- Visual editor for VSARL files
- Syntax highlighting
- Run button to execute programs
- Console output for results and errors
- Integrated query interface
- KB statistics viewer

## Solution Summary
Build a Python-based IDE using Tkinter (built into Python, no extra dependencies) with:
- **Editor pane**: Syntax-highlighted text editor for VSARL code
- **Console pane**: Output display for results, errors, and status messages
- **Toolbar**: Quick access to File, Run, Query, Stats operations
- **Menu bar**: Full menu system with keyboard shortcuts
- **Status bar**: Current file, line/column, execution status

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  File  Edit  Run  Help                               │  ← Menu
├─────────────────────────────────────────────────────┤
│  📄 New  📁 Open  💾 Save  ▶️ Run  🔍 Query  📊 Stats │  ← Toolbar
├───────────────────────┬─────────────────────────────┤
│                       │                             │
│  Editor Pane          │  Console Pane               │
│  (VSARL code with     │  (Results, errors,          │
│   syntax highlighting)│   status messages)          │
│                       │                             │
│                       │                             │
│                       │                             │
│                       │                             │
├───────────────────────┴─────────────────────────────┤
│  File: untitled.vsar  │  Line: 1  Col: 1  │  Ready │  ← Status
└─────────────────────────────────────────────────────┘
```

## Implementation Plan

### Stage 1: Project Structure & Basic Window ✓/✗
- [ ] Create `src/vsar/ide/` package directory
- [ ] Create `src/vsar/ide/__init__.py`
- [ ] Create `src/vsar/ide/main.py` - Main IDE application window
- [ ] Add `vsar-ide` entry point to pyproject.toml
- [ ] Test: Can launch IDE window with basic title

**Files to create:**
- `src/vsar/ide/__init__.py`
- `src/vsar/ide/main.py`

**Files to modify:**
- `pyproject.toml` - Add vsar-ide script entry

### Stage 2: Layout & Widgets ✓/✗
- [ ] Create split pane layout (PanedWindow)
- [ ] Add editor widget (ScrolledText) on left
- [ ] Add console widget (ScrolledText) on right, read-only
- [ ] Add menu bar with File, Edit, Run, Help menus
- [ ] Add toolbar with icon buttons
- [ ] Add status bar at bottom
- [ ] Test: Can type in editor, see console area

**Files to create:**
- `src/vsar/ide/widgets.py` - Custom widgets (editor, console)

### Stage 3: Syntax Highlighting ✓/✗
- [ ] Create `src/vsar/ide/highlighter.py`
- [ ] Define VSARL syntax patterns (facts, rules, queries, directives, comments)
- [ ] Implement syntax highlighting with text tags
- [ ] Apply highlighting on text changes (with debouncing)
- [ ] Test: Keywords, comments, strings highlighted correctly

**Files to create:**
- `src/vsar/ide/highlighter.py`

**Colors:**
- Keywords (fact, rule, query): blue
- Directives (@model, @beam): purple
- Comments (// and /* */): gray
- Strings: green
- Variables (uppercase): orange
- Predicates: black

### Stage 4: File Operations ✓/✗
- [ ] Implement New File (Ctrl+N)
- [ ] Implement Open File (Ctrl+O) with file dialog
- [ ] Implement Save File (Ctrl+S)
- [ ] Implement Save As (Ctrl+Shift+S)
- [ ] Track current file path and modified state
- [ ] Show modified indicator (*) in title bar
- [ ] Confirm before closing with unsaved changes
- [ ] Test: Can create, open, save, close files

**Files to modify:**
- `src/vsar/ide/main.py` - Add file operation methods

### Stage 5: Run Program Integration ✓/✗
- [ ] Create `src/vsar/ide/runner.py` - Program execution backend
- [ ] Implement Run button handler
- [ ] Parse program from editor text
- [ ] Execute program using VSAREngine
- [ ] Display results in console pane with formatting
- [ ] Display errors in console pane (red text)
- [ ] Add progress indicator during execution
- [ ] Test: Can run example programs, see results

**Files to create:**
- `src/vsar/ide/runner.py`

**Integration:**
- Use existing `vsar.language.parser.parse()`
- Use existing `vsar.semantics.engine.VSAREngine`
- Format output similar to CLI formatters

### Stage 6: Query Interface ✓/✗
- [ ] Add "Run Query" dialog (Ctrl+Q)
- [ ] Text entry for query input
- [ ] Execute query against current KB
- [ ] Display query results in console
- [ ] Handle errors gracefully
- [ ] Test: Can run ad-hoc queries after loading program

**Files to modify:**
- `src/vsar/ide/main.py` - Add query dialog

### Stage 7: KB Statistics Viewer ✓/✗
- [ ] Add "Show Stats" button/menu item
- [ ] Display KB statistics in console or dialog
- [ ] Show total facts, predicates, counts
- [ ] Test: Can view KB stats after loading program

**Files to modify:**
- `src/vsar/ide/main.py` - Add stats display

### Stage 8: Console Improvements ✓/✗
- [ ] Add colored output (info, error, success)
- [ ] Add Clear Console button
- [ ] Add timestamp to messages
- [ ] Make console scrollable
- [ ] Auto-scroll to bottom on new output
- [ ] Test: Console displays nicely formatted output

### Stage 9: Polish & UX ✓/✗
- [ ] Add keyboard shortcuts reference (Help menu)
- [ ] Add About dialog
- [ ] Add example programs menu (load examples)
- [ ] Set window icon
- [ ] Set proper window title format
- [ ] Handle window close event properly
- [ ] Test: All UI elements work smoothly

### Stage 10: Documentation & Testing ✓/✗
- [ ] Create `docs/ide-guide.md` - User guide for IDE
- [ ] Update README.md with IDE section
- [ ] Add screenshots to documentation
- [ ] Create basic tests for IDE components
- [ ] Update CHANGELOG.md
- [ ] Test: IDE works on Windows, Linux, macOS

## Success Criteria
- [ ] Can launch IDE with `vsar-ide` command
- [ ] Can create, open, save VSARL files
- [ ] Syntax highlighting works correctly
- [ ] Can run programs and see results
- [ ] Can execute queries interactively
- [ ] Can view KB statistics
- [ ] Console displays formatted output
- [ ] All keyboard shortcuts work
- [ ] Cross-platform compatible

## Key Technologies
- **Tkinter**: GUI framework (built into Python)
- **tkinter.scrolledtext**: Text editor widget
- **tkinter.filedialog**: File open/save dialogs
- **tkinter.messagebox**: Alerts and confirmations
- **Existing VSAR library**: Parser, engine, formatters

## Design Principles
- **Simple and focused**: Not trying to be a full IDE like VS Code
- **Minimal dependencies**: Only use Python standard library + existing vsar deps
- **Integration**: Reuse existing VSAR components (parser, engine, formatters)
- **Cross-platform**: Works on Windows, Linux, macOS
- **Educational**: Similar to DrRacket - great for learning VSAR

## Estimated Effort
- Stage 1-2: 1-2 hours (basic window and layout)
- Stage 3: 2-3 hours (syntax highlighting - requires pattern work)
- Stage 4: 1 hour (file operations)
- Stage 5: 2-3 hours (program execution integration)
- Stage 6: 1 hour (query interface)
- Stage 7: 30 min (stats viewer)
- Stage 8: 1 hour (console improvements)
- Stage 9: 1-2 hours (polish)
- Stage 10: 1 hour (documentation)
- **Total: ~12-15 hours** (1-2 days of focused work)

## Notes
- Keep UI simple and clean - not trying to compete with VS Code
- Focus on educational use case - great for learning VSAR
- Syntax highlighting can be basic at first (improve later)
- Reuse existing CLI formatters for output
- Add more features incrementally based on user feedback
