# Makefile for VSAR paper

# Main tex file (without .tex extension)
MAIN = vsar-encoding

# Output PDF
PDF = $(MAIN).pdf

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# LaTeX flags
LATEXFLAGS = -interaction=nonstopmode

# Default target
.PHONY: all
all: $(PDF)

# Build PDF with bibliography
$(PDF): $(MAIN).tex references.bib
	@echo "Building $(PDF)..."
	@$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@$(BIBTEX) $(MAIN)
	@$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "✓ Build complete: $(PDF)"

# Quick build (no bibliography update)
.PHONY: quick
quick:
	@echo "Quick build (no bibtex)..."
	@$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Clean intermediate files
.PHONY: clean
clean:
	@echo "Cleaning intermediate files..."
	@rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot

# Clean everything including PDF
.PHONY: distclean
distclean: clean
	@echo "Cleaning PDF..."
	@rm -f $(PDF)

# View PDF (platform-specific)
.PHONY: view
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	elif command -v start > /dev/null; then \
		start $(PDF); \
	else \
		echo "Cannot detect PDF viewer. Please open $(PDF) manually."; \
	fi

# Continuous build (requires latexmk)
.PHONY: watch
watch:
	@if command -v latexmk > /dev/null; then \
		latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex; \
	else \
		echo "Error: latexmk not found. Install with: tlmgr install latexmk"; \
		exit 1; \
	fi

# Help
.PHONY: help
help:
	@echo "VSAR Paper Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make           - Build PDF with bibliography (default)"
	@echo "  make quick     - Quick build without updating bibliography"
	@echo "  make clean     - Remove intermediate files"
	@echo "  make distclean - Remove all generated files including PDF"
	@echo "  make view      - Build and open PDF in viewer"
	@echo "  make watch     - Continuous build on file changes (requires latexmk)"
	@echo "  make help      - Show this help message"
