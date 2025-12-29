#!/bin/bash
# Version bumping and release script

set -e

# Check if version argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./scripts/bump_version.sh <version>"
    echo "Example: ./scripts/bump_version.sh 0.2.0"
    exit 1
fi

NEW_VERSION=$1

# Validate version format (basic check)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 0.2.0)"
    exit 1
fi

echo "Bumping version to $NEW_VERSION..."

# Update version in pyproject.toml
sed -i "s/^version = .*/version = \"$NEW_VERSION\"/" pyproject.toml

# Update version in src/vsar/version.py
echo "__version__ = \"$NEW_VERSION\"" > src/vsar/version.py

echo "✓ Updated version to $NEW_VERSION"
echo ""
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Commit: git add -A && git commit -m 'Bump version to $NEW_VERSION'"
echo "3. Tag: git tag v$NEW_VERSION"
echo "4. Push: git push origin main --tags"
echo ""
echo "GitHub Actions will automatically:"
echo "  - Run tests"
echo "  - Build package"
echo "  - Publish to PyPI"
echo "  - Create GitHub Release"
