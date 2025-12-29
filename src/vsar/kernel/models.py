"""Model configurations for VSAR kernel backends wrapping VSAX."""

from dataclasses import dataclass

from vsax import VSAModel


@dataclass
class VSARModelConfig:
    """Configuration for VSAR kernel backends."""

    dim: int
    """Hypervector dimension."""

    seed: int = 42
    """Random seed for reproducibility."""


@dataclass
class FHRRConfig(VSARModelConfig):
    """Configuration for FHRR (Fourier Holographic Reduced Representations) backend."""

    pass


@dataclass
class MAPConfig(VSARModelConfig):
    """Configuration for MAP (Multiply-Add-Permute) backend."""

    pass


@dataclass
class CliffordConfig(VSARModelConfig):
    """Configuration for Clifford algebra backend."""

    pass
