# Release Process

This document describes how to create a new release of VSAR.

## Automated Publishing Setup

VSAR uses GitHub Actions to automatically publish to PyPI when you push a version tag.

### One-Time Setup: PyPI Trusted Publishing

1. **Go to PyPI**: https://pypi.org/manage/account/publishing/
2. **Add a new pending publisher** with:
   - **PyPI Project Name**: `vsar`
   - **Owner**: `vasanthsarathy`
   - **Repository name**: `vsar`
   - **Workflow name**: `publish.yml`
   - **Environment name**: (leave blank)

3. **That's it!** No API tokens needed. GitHub will authenticate using OIDC.

## Release Workflow

### Method 1: Using the Helper Script (Recommended)

```bash
# Bump version (updates pyproject.toml and version.py)
./scripts/bump_version.sh 0.2.0

# Review changes
git diff

# Commit and tag
git add -A
git commit -m "Bump version to 0.2.0"
git tag v0.2.0

# Push (triggers automated publishing)
git push origin main --tags
```

### Method 2: Manual Process

1. **Update version in `pyproject.toml`:**
   ```toml
   [project]
   version = "0.2.0"
   ```

2. **Update version in `src/vsar/version.py`:**
   ```python
   __version__ = "0.2.0"
   ```

3. **Commit, tag, and push:**
   ```bash
   git add pyproject.toml src/vsar/version.py
   git commit -m "Bump version to 0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```

## What Happens Automatically

When you push a tag (e.g., `v0.2.0`), GitHub Actions will:

1. ✅ **Run full test suite** (must pass with ≥90% coverage)
2. ✅ **Build package** (`uv build`)
3. ✅ **Publish to PyPI** (using Trusted Publishing)
4. ✅ **Create GitHub Release** (with release notes and artifacts)

You can monitor progress at: https://github.com/vasanthsarathy/vsar/actions

## Version Numbering

VSAR follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.2.0): New functionality, backwards compatible
- **PATCH** version (0.1.1): Bug fixes, backwards compatible

### Current Phase Guidelines

- **Phase 0 (Foundation)**: v0.1.x
- **Phase 1 (Language/CLI)**: v0.2.x
- **Phase 2 (Rules)**: v0.3.x
- **Phase 3 (Optimization)**: v0.4.x
- **Production ready**: v1.0.0

## Pre-Release Checklist

Before creating a release, ensure:

- [ ] All tests passing: `uv run pytest`
- [ ] Coverage ≥90%: `uv run pytest --cov=vsar --cov-fail-under=90`
- [ ] Linting passes: `uv run ruff check .`
- [ ] Formatting applied: `uv run black .`
- [ ] Documentation updated (README, CHANGELOG)
- [ ] Breaking changes documented

## Troubleshooting

### Publishing Fails

**Problem**: "Trusted publishing exchange failure"
**Solution**: Verify PyPI trusted publisher settings match exactly:
- Workflow: `publish.yml`
- Repo: `vasanthsarathy/vsar`

**Problem**: Tests fail in CI
**Solution**: Run locally first: `uv run pytest --cov=vsar --cov-fail-under=90`

### Tag Already Exists

```bash
# Delete local tag
git tag -d v0.2.0

# Delete remote tag
git push origin :refs/tags/v0.2.0

# Recreate and push
git tag v0.2.0
git push origin v0.2.0
```

## Post-Release

After successful release:

1. **Verify on PyPI**: https://pypi.org/project/vsar/
2. **Test installation**: `pip install vsar==0.2.0`
3. **Announce** on relevant channels
4. **Update documentation** if needed

## Emergency Rollback

If a release has critical issues:

```bash
# Yank the bad release (doesn't delete, marks as unavailable)
pip install twine
twine upload --repository pypi --skip-existing dist/*

# Then visit PyPI and click "Yank" on the bad version
```

Users can still install yanked versions explicitly, but won't get them by default.
