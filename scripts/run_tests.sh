#!/bin/bash
# Test runner script with different modes

set -e

MODE="${1:-quick}"

case "$MODE" in
  quick)
    echo "Running quick tests (language + CLI)..."
    uv run pytest tests/unit/language tests/unit/cli -v
    ;;

  unit)
    echo "Running all unit tests..."
    uv run pytest tests/unit/ -v
    ;;

  integration)
    echo "Running integration tests..."
    uv run pytest tests/integration/ -v
    ;;

  coverage)
    echo "Running tests with coverage..."
    uv run pytest --cov=vsar --cov-report=html --cov-report=term-missing
    ;;

  ci)
    echo "Running CI tests (full suite with coverage)..."
    uv run pytest --cov=vsar --cov-fail-under=90
    ;;

  failed)
    echo "Re-running only failed tests..."
    uv run pytest --lf -v
    ;;

  *)
    echo "Usage: $0 {quick|unit|integration|coverage|ci|failed}"
    echo ""
    echo "Modes:"
    echo "  quick       - Fast sanity check (language + CLI tests)"
    echo "  unit        - All unit tests"
    echo "  integration - Integration tests only"
    echo "  coverage    - Tests with coverage report"
    echo "  ci          - Full CI test suite (with 90% coverage requirement)"
    echo "  failed      - Re-run only previously failed tests"
    exit 1
    ;;
esac
