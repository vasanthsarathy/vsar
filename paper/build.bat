@echo off
REM Build script for VSAR paper (Windows)

echo Building VSAR paper...

REM Check if pdflatex is installed
where pdflatex >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: pdflatex not found. Please install MiKTeX or TeX Live.
    exit /b 1
)

REM Check if bibtex is installed
where bibtex >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: bibtex not found. Please install MiKTeX or TeX Live.
    exit /b 1
)

REM Clean old build files
echo Cleaning old build files...
del /Q *.aux *.log *.bbl *.blg *.out *.toc *.pdf 2>nul

REM First pass - generate aux files
echo Running pdflatex (pass 1)...
pdflatex -interaction=nonstopmode vsar-encoding.tex
if %ERRORLEVEL% NEQ 0 (
    echo Error in first pdflatex pass
    exit /b 1
)

REM Run bibtex to process references
echo Running bibtex...
bibtex vsar-encoding
if %ERRORLEVEL% NEQ 0 (
    echo Error in bibtex
    exit /b 1
)

REM Second pass - incorporate references
echo Running pdflatex (pass 2)...
pdflatex -interaction=nonstopmode vsar-encoding.tex
if %ERRORLEVEL% NEQ 0 (
    echo Error in second pdflatex pass
    exit /b 1
)

REM Third pass - fix references
echo Running pdflatex (pass 3)...
pdflatex -interaction=nonstopmode vsar-encoding.tex
if %ERRORLEVEL% NEQ 0 (
    echo Error in third pdflatex pass
    exit /b 1
)

REM Clean intermediate files
echo Cleaning intermediate files...
del /Q *.aux *.log *.bbl *.blg *.out *.toc 2>nul

echo.
echo Build complete! Output: vsar-encoding.pdf
echo.
