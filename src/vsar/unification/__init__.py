"""Unification kernel for VSAR - structure-aware decoding and pattern matching."""

from .substitution import Substitution
from .decoder import StructureDecoder
from .unifier import Unifier

__all__ = ["Substitution", "StructureDecoder", "Unifier"]
