"""Pytest configuration and shared fixtures for VSAR tests."""

import pytest


@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducible tests."""
    return 42
