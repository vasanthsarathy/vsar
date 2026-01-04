# VSAR Paper - Build Instructions

This directory contains the LaTeX source for the VSAR journal paper.

## Prerequisites

You need a LaTeX distribution installed:

### Windows
- **MiKTeX**: https://miktex.org/download
- **TeX Live**: https://www.tug.org/texlive/

### Mac
```bash
brew install --cask mactex
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install texlive-full
```

### Linux (Fedora/RHEL)
```bash
sudo dnf install texlive-scheme-full
```

## Building the PDF

### Option 1: Use the build script (Recommended)

**Windows:**
```cmd
cd paper
build.bat
```

**Unix/Mac/Linux:**
```bash
cd paper
chmod +x build.sh
./build.sh
```

### Option 2: Use Make

If you have `make` installed:

```bash
cd paper
make          # Build PDF
make view     # Build and open PDF
make clean    # Clean intermediate files
make watch    # Continuous build (requires latexmk)
```

### Option 3: Manual compilation

```bash
cd paper
pdflatex vsar-encoding.tex
bibtex vsar-encoding
pdflatex vsar-encoding.tex
pdflatex vsar-encoding.tex
```

## Output

The build process creates:
- **`vsar-encoding.pdf`** - The compiled paper

Intermediate files (`.aux`, `.log`, `.bbl`, etc.) are automatically cleaned.

## Files

- `vsar-encoding.tex` - Main LaTeX source
- `references.bib` - Bibliography (BibTeX format)
- `ROADMAP.md` - Development roadmap for completing the paper
- `KEY_INSIGHTS.md` - Mathematical insights and proof strategies

## Troubleshooting

### Missing LaTeX packages

If you get errors about missing packages:

**MiKTeX (Windows):**
- MiKTeX will prompt to auto-install missing packages
- Or manually: Open MiKTeX Console → Packages → Install

**TeX Live (Mac/Linux):**
```bash
# Install missing package (e.g., algorithmicx)
tlmgr install algorithmicx
```

### Common packages used
- `amsmath`, `amssymb`, `amsthm` - Math symbols and theorems
- `algorithm`, `algorithmic` - Algorithm formatting
- `graphicx` - Figures and images
- `hyperref` - Hyperlinks and PDF bookmarks
- `booktabs` - Professional tables

### Build errors

If the build fails:

1. Check the `.log` file for specific errors
2. Ensure all packages are installed
3. Try cleaning and rebuilding:
   ```bash
   make distclean
   make
   ```

### Font issues

If you get font warnings, install these packages:

```bash
# TeX Live
tlmgr install collection-fontsrecommended

# Ubuntu/Debian
sudo apt-get install texlive-fonts-recommended
```

## Development Workflow

1. Edit `vsar-encoding.tex`
2. Run `make` to build
3. View `vsar-encoding.pdf`
4. Iterate

For continuous editing, use:
```bash
make watch  # Auto-rebuilds on file changes
```

## Contributing

When adding content:

1. Follow existing LaTeX style and notation
2. Use consistent theorem numbering
3. Add new references to `references.bib`
4. Test build before committing

## Next Steps

See `ROADMAP.md` for sections that need expansion and the development timeline.

## Questions?

Check the TeX error log or consult:
- TeX Stack Exchange: https://tex.stackexchange.com/
- LaTeX Wikibook: https://en.wikibooks.org/wiki/LaTeX
