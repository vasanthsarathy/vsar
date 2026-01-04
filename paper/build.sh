#!/bin/bash
# Build script for VSAR paper (Unix/Mac/Linux)

set -e  # Exit on error

echo "Building VSAR paper..."

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Please install TeX Live or MiKTeX."
    exit 1
fi

# Check if bibtex is installed
if ! command -v bibtex &> /dev/null; then
    echo "Error: bibtex not found. Please install TeX Live or MiKTeX."
    exit 1
fi

# Clean old build files
echo "Cleaning old build files..."
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.pdf

# First pass - generate aux files
echo "Running pdflatex (pass 1)..."
pdflatex -interaction=nonstopmode vsar-encoding.tex

# Run bibtex to process references
echo "Running bibtex..."
bibtex vsar-encoding

# Second pass - incorporate references
echo "Running pdflatex (pass 2)..."
pdflatex -interaction=nonstopmode vsar-encoding.tex

# Third pass - fix references
echo "Running pdflatex (pass 3)..."
pdflatex -interaction=nonstopmode vsar-encoding.tex

# Clean intermediate files
echo "Cleaning intermediate files..."
rm -f *.aux *.log *.bbl *.blg *.out *.toc

echo ""
echo "✓ Build complete! Output: vsar-encoding.pdf"
echo ""
