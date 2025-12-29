"""VSA backend implementations wrapping VSAX models."""

import jax
import jax.numpy as jnp
from vsax import (
    VSAModel,
    cosine_similarity,
    create_fhrr_model,
    create_map_model,
    sample_complex_random,
    sample_random,
)

from .base import KernelBackend
from .models import FHRRConfig, MAPConfig


class FHRRBackend(KernelBackend):
    """
    FHRR (Fourier Holographic Reduced Representations) backend.

    This backend uses VSAX's FHRR model, which employs FFT-based circular
    convolution for binding operations. FHRR uses complex-valued hypervectors.

    Args:
        dim: Hypervector dimension
        seed: Random seed for reproducibility

    Example:
        >>> backend = FHRRBackend(dim=512, seed=42)
        >>> key = jax.random.PRNGKey(0)
        >>> a = backend.generate_random(key, (512,))
        >>> b = backend.generate_random(jax.random.split(key)[0], (512,))
        >>> c = backend.bind(a, b)
        >>> b_recovered = backend.unbind(c, a)
        >>> sim = backend.similarity(b, b_recovered)
        >>> assert sim > 0.9  # High similarity for good reconstruction
    """

    def __init__(self, dim: int = 512, seed: int = 42):
        self._dim = dim
        self._seed = seed
        self._model: VSAModel = create_fhrr_model(dim=dim)
        self._rng = jax.random.PRNGKey(seed)

    def bind(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Bind two hypervectors using circular convolution (via FFT).

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Bound hypervector
        """
        # VSAX FHRR uses circular convolution for binding
        return self._model.opset.bind(a, b)

    def unbind(self, bound: jnp.ndarray, factor: jnp.ndarray) -> jnp.ndarray:
        """
        Unbind using complex conjugate and normalization.

        For FHRR: inverse(a)[k] = a[k] / |a[k]|^2 (complex conjugate + normalization)
        Then unbind(c, a) = bind(c, inverse(a))

        Args:
            bound: Bound hypervector
            factor: Known factor to remove

        Returns:
            Unbound hypervector (approximate)
        """
        # VSAX inverse performs: a^(-1)[k] = a[k] / |a[k]|^2
        factor_inv = self._model.opset.inverse(factor)
        result = self._model.opset.bind(bound, factor_inv)
        return self.normalize(result)

    def bundle(
        self, vectors: list[jnp.ndarray] | jnp.ndarray, axis: int = 0
    ) -> jnp.ndarray:
        """
        Bundle multiple hypervectors via element-wise sum and normalization.

        Args:
            vectors: List of hypervectors or array of hypervectors
            axis: Axis along which to bundle (ignored for VSAX, kept for interface compatibility)

        Returns:
            Bundled hypervector
        """
        if isinstance(vectors, list):
            # VSAX bundle takes variable args (*vecs)
            bundled = self._model.opset.bundle(*vectors)
        else:
            # If array, unpack along first axis
            bundled = self._model.opset.bundle(*list(vectors))
        return self.normalize(bundled)

    def similarity(self, a: jnp.ndarray, b: jnp.ndarray) -> float:
        """
        Compute cosine similarity between two hypervectors.

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Similarity score in [0, 1]
        """
        # VSAX provides cosine_similarity
        sim = cosine_similarity(a, b)
        # Normalize to [0, 1] range (cosine is in [-1, 1])
        return float((sim + 1.0) / 2.0)

    def generate_random(
        self, key: jax.random.PRNGKey, shape: tuple[int, ...]
    ) -> jnp.ndarray:
        """
        Generate random hypervector using VSAX's sampling function.

        Args:
            key: JAX random key
            shape: Shape of the vector (should be (dim,))

        Returns:
            Random complex hypervector
        """
        # VSAX sample_complex_random returns (n, dim) shape
        # We squeeze to get (dim,) for single vector
        vec = sample_complex_random(dim=self._dim, n=1, key=key)
        return vec.squeeze(axis=0)

    @property
    def dimension(self) -> int:
        """Return hypervector dimension."""
        return self._dim

    def normalize(self, vec: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize a hypervector.

        For FHRR (complex vectors), normalization ensures unit magnitude.

        Args:
            vec: Hypervector to normalize

        Returns:
            Normalized hypervector
        """
        # For complex hypervectors, normalize to unit magnitude
        norm = jnp.linalg.norm(vec)
        return vec / (norm + 1e-10)  # Add epsilon to avoid division by zero


class MAPBackend(KernelBackend):
    """
    MAP (Multiply-Add-Permute) backend.

    This backend uses VSAX's MAP model, which employs element-wise multiplication
    for binding. MAP uses real-valued hypervectors.

    Args:
        dim: Hypervector dimension
        seed: Random seed for reproducibility

    Example:
        >>> backend = MAPBackend(dim=512, seed=42)
        >>> key = jax.random.PRNGKey(0)
        >>> a = backend.generate_random(key, (512,))
        >>> b = backend.generate_random(jax.random.split(key)[0], (512,))
        >>> c = backend.bind(a, b)
    """

    def __init__(self, dim: int = 512, seed: int = 42):
        self._dim = dim
        self._seed = seed
        self._model: VSAModel = create_map_model(dim=dim)
        self._rng = jax.random.PRNGKey(seed)

    def bind(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Bind two hypervectors using element-wise multiplication.

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Bound hypervector
        """
        return self._model.opset.bind(a, b)

    def unbind(self, bound: jnp.ndarray, factor: jnp.ndarray) -> jnp.ndarray:
        """
        Unbind using inverse operation for MAP.

        For MAP: unbind(c, a) = bind(c, inverse(a))
        This recovers b from c = bind(a, b)

        Args:
            bound: Bound hypervector
            factor: Known factor to remove

        Returns:
            Unbound hypervector
        """
        # MAP unbinding uses inverse + bind
        factor_inv = self._model.opset.inverse(factor)
        result = self._model.opset.bind(bound, factor_inv)
        return self.normalize(result)

    def bundle(
        self, vectors: list[jnp.ndarray] | jnp.ndarray, axis: int = 0
    ) -> jnp.ndarray:
        """
        Bundle multiple hypervectors via element-wise sum and normalization.

        Args:
            vectors: List of hypervectors or array of hypervectors
            axis: Axis along which to bundle (ignored for VSAX, kept for interface compatibility)

        Returns:
            Bundled hypervector
        """
        if isinstance(vectors, list):
            # VSAX bundle takes variable args (*vecs)
            bundled = self._model.opset.bundle(*vectors)
        else:
            # If array, unpack along first axis
            bundled = self._model.opset.bundle(*list(vectors))
        return self.normalize(bundled)

    def similarity(self, a: jnp.ndarray, b: jnp.ndarray) -> float:
        """
        Compute cosine similarity between two hypervectors.

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Similarity score in [0, 1]
        """
        sim = cosine_similarity(a, b)
        return float((sim + 1.0) / 2.0)

    def generate_random(
        self, key: jax.random.PRNGKey, shape: tuple[int, ...]
    ) -> jnp.ndarray:
        """
        Generate random hypervector using VSAX's sampling function.

        Args:
            key: JAX random key
            shape: Shape of the vector (should be (dim,))

        Returns:
            Random real hypervector
        """
        # VSAX sample_random returns (n, dim) shape
        # We squeeze to get (dim,) for single vector
        vec = sample_random(dim=self._dim, n=1, key=key)
        return vec.squeeze(axis=0)

    @property
    def dimension(self) -> int:
        """Return hypervector dimension."""
        return self._dim

    def normalize(self, vec: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize a hypervector to unit L2 norm.

        Args:
            vec: Hypervector to normalize

        Returns:
            Normalized hypervector
        """
        norm = jnp.linalg.norm(vec)
        return vec / (norm + 1e-10)
