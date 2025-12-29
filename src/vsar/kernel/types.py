"""Type definitions for VSAR kernel."""

from enum import Enum
from typing import Protocol

import jax.numpy as jnp


class BackendType(Enum):
    """Supported backend types."""

    FHRR = "fhrr"  # Fourier Holographic Reduced Representations
    MAP = "map"  # Multiply-Add-Permute
    CLIFFORD = "clifford"  # Clifford algebra


class Vector(Protocol):
    """Protocol for hypervector types."""

    shape: tuple[int, ...]
    dtype: jnp.dtype
