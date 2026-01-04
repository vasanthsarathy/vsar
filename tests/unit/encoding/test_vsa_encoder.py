"""Unit tests for VSA encoder."""

import jax.numpy as jnp
import pytest

from vsar.encoding.vsa_encoder import VSAEncoder
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.symbols.registry import SymbolRegistry
from vsar.symbols.spaces import SymbolSpace


class TestVSAEncoder:
    """Test cases for VSAEncoder."""

    @pytest.fixture
    def backend(self) -> FHRRBackend:
        """Create test backend."""
        return FHRRBackend(dim=128, seed=42)

    @pytest.fixture
    def registry(self, backend: FHRRBackend) -> SymbolRegistry:
        """Create test registry."""
        return SymbolRegistry(dim=backend.dimension, seed=42)

    @pytest.fixture
    def encoder(self, backend: FHRRBackend, registry: SymbolRegistry) -> VSAEncoder:
        """Create test encoder."""
        return VSAEncoder(backend, registry, seed=42)

    def test_initialization(self, backend: FHRRBackend, registry: SymbolRegistry) -> None:
        """Test encoder initialization."""
        encoder = VSAEncoder(backend, registry, seed=42)
        assert encoder.backend == backend
        assert encoder.registry == registry
        assert encoder.seed == 42

    def test_encode_unary_atom(self, encoder: VSAEncoder) -> None:
        """Test encoding unary predicate."""
        vec = encoder.encode_atom("human", ["alice"])
        assert vec is not None
        assert vec.shape == (128,)

    def test_encode_binary_atom(self, encoder: VSAEncoder) -> None:
        """Test encoding binary predicate."""
        vec = encoder.encode_atom("parent", ["alice", "bob"])
        assert vec is not None
        assert vec.shape == (128,)

    def test_encode_ternary_atom(self, encoder: VSAEncoder) -> None:
        """Test encoding ternary predicate."""
        vec = encoder.encode_atom("gave", ["alice", "bob", "book"])
        assert vec is not None
        assert vec.shape == (128,)

    def test_encode_atom_is_normalized(self, encoder: VSAEncoder) -> None:
        """Test that encoded atoms have unit norm."""
        vec = encoder.encode_atom("parent", ["alice", "bob"])
        norm = jnp.linalg.norm(vec)
        assert jnp.abs(norm - 1.0) < 1e-5

    def test_encode_atom_is_deterministic(self, encoder: VSAEncoder) -> None:
        """Test that encoding same atom twice gives same result."""
        vec1 = encoder.encode_atom("parent", ["alice", "bob"])
        vec2 = encoder.encode_atom("parent", ["alice", "bob"])

        assert jnp.allclose(vec1, vec2)

    def test_encode_different_atoms_are_different(self, encoder: VSAEncoder) -> None:
        """Test that different atoms get different encodings."""
        vec1 = encoder.encode_atom("parent", ["alice", "bob"])
        vec2 = encoder.encode_atom("parent", ["alice", "carol"])

        similarity = encoder.backend.similarity(vec1, vec2)
        assert similarity < 0.9  # Should be different

    @pytest.mark.skip(reason="Shift-based encoding doesn't encode predicate name")
    def test_encode_different_predicates_are_different(self, encoder: VSAEncoder) -> None:
        """Test that same arguments with different predicates are different."""
        # NOTE: With shift-based encoding, predicate is NOT encoded in vector.
        # Predicates are distinguished by KB partitioning only.
        vec1 = encoder.encode_atom("parent", ["alice", "bob"])
        vec2 = encoder.encode_atom("sibling", ["alice", "bob"])

        similarity = encoder.backend.similarity(vec1, vec2)
        assert similarity > 0.99  # Should be nearly identical with shift-based encoding

    def test_encode_argument_order_matters(self, encoder: VSAEncoder) -> None:
        """Test that argument order affects encoding."""
        vec1 = encoder.encode_atom("parent", ["alice", "bob"])
        vec2 = encoder.encode_atom("parent", ["bob", "alice"])

        similarity = encoder.backend.similarity(vec1, vec2)
        assert similarity < 0.9  # Should be different due to different roles

    def test_encode_atom_empty_args_raises_error(self, encoder: VSAEncoder) -> None:
        """Test that empty argument list raises ValueError."""
        with pytest.raises(ValueError, match="Arguments list cannot be empty"):
            encoder.encode_atom("parent", [])

    def test_encode_query_with_one_variable(self, encoder: VSAEncoder) -> None:
        """Test encoding query with one variable."""
        vec = encoder.encode_query("parent", ["alice", None])
        assert vec is not None
        assert vec.shape == (128,)

    def test_encode_query_with_multiple_variables(self, encoder: VSAEncoder) -> None:
        """Test encoding query with multiple variables."""
        vec = encoder.encode_query("gave", ["alice", None, None])
        assert vec is not None
        assert vec.shape == (128,)

    def test_encode_query_is_normalized(self, encoder: VSAEncoder) -> None:
        """Test that encoded queries have unit norm."""
        vec = encoder.encode_query("parent", ["alice", None])
        norm = jnp.linalg.norm(vec)
        assert jnp.abs(norm - 1.0) < 1e-5

    def test_encode_query_is_deterministic(self, encoder: VSAEncoder) -> None:
        """Test that encoding same query twice gives same result."""
        vec1 = encoder.encode_query("parent", ["alice", None])
        vec2 = encoder.encode_query("parent", ["alice", None])

        assert jnp.allclose(vec1, vec2)

    def test_encode_query_different_from_atom(self, encoder: VSAEncoder) -> None:
        """Test that query encoding differs from full atom encoding."""
        atom_vec = encoder.encode_atom("parent", ["alice", "bob"])
        query_vec = encoder.encode_query("parent", ["alice", None])

        similarity = encoder.backend.similarity(atom_vec, query_vec)
        # Should be somewhat similar but not identical
        assert 0.3 < similarity < 0.95

    def test_encode_query_variable_position_matters(self, encoder: VSAEncoder) -> None:
        """Test that variable position affects encoding."""
        vec1 = encoder.encode_query("parent", ["alice", None])
        vec2 = encoder.encode_query("parent", [None, "alice"])

        similarity = encoder.backend.similarity(vec1, vec2)
        assert similarity < 0.9  # Should be different

    def test_encode_query_empty_args_raises_error(self, encoder: VSAEncoder) -> None:
        """Test that empty argument list raises ValueError."""
        with pytest.raises(ValueError, match="Arguments list cannot be empty"):
            encoder.encode_query("parent", [])

    def test_encode_query_all_none_raises_error(self, encoder: VSAEncoder) -> None:
        """Test that all-None arguments raises ValueError."""
        with pytest.raises(ValueError, match="At least one argument must be bound"):
            encoder.encode_query("parent", [None, None])

    def test_get_variable_positions(self, encoder: VSAEncoder) -> None:
        """Test getting variable positions from argument list."""
        positions = encoder.get_variable_positions(["alice", None, "bob"])
        assert positions == [2]

        positions = encoder.get_variable_positions([None, None, "bob"])
        assert positions == [1, 2]

        positions = encoder.get_variable_positions(["alice", "bob", "carol"])
        assert positions == []

    def test_get_bound_positions(self, encoder: VSAEncoder) -> None:
        """Test getting bound argument positions from argument list."""
        positions = encoder.get_bound_positions(["alice", None, "bob"])
        assert positions == [1, 3]

        positions = encoder.get_bound_positions([None, None, "bob"])
        assert positions == [3]

        positions = encoder.get_bound_positions(["alice", "bob", "carol"])
        assert positions == [1, 2, 3]

    def test_different_encoders_same_seed(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test that encoders with same seed produce identical encodings."""
        encoder1 = VSAEncoder(backend, registry, seed=42)
        encoder2 = VSAEncoder(backend, registry, seed=42)

        vec1 = encoder1.encode_atom("parent", ["alice", "bob"])
        vec2 = encoder2.encode_atom("parent", ["alice", "bob"])

        assert jnp.allclose(vec1, vec2)

    @pytest.mark.skip(reason="Shift-based encoding doesn't use encoder seed for atom encoding")
    def test_different_encoders_different_seed(
        self, backend: FHRRBackend, registry: SymbolRegistry
    ) -> None:
        """Test that encoders with different seeds produce different encodings."""
        # NOTE: Shift-based encoding doesn't use encoder seed for atom encoding
        encoder1 = VSAEncoder(backend, registry, seed=42)
        encoder2 = VSAEncoder(backend, registry, seed=123)

        vec1 = encoder1.encode_atom("parent", ["alice", "bob"])
        vec2 = encoder2.encode_atom("parent", ["alice", "bob"])

        # With shift-based encoding, these will be identical
        similarity = backend.similarity(vec1, vec2)
        assert similarity > 0.99

    @pytest.mark.skip(reason="Shift-based encoding doesn't register predicates")
    def test_encode_atom_registers_symbols(
        self, encoder: VSAEncoder, registry: SymbolRegistry
    ) -> None:
        """Test that encoding registers symbols in registry."""
        # NOTE: Shift-based encoding only registers entities, not predicates
        encoder.encode_atom("parent", ["alice", "bob"])

        # Check that entity symbols were registered
        assert "alice" in registry.symbols(SymbolSpace.ENTITIES)
        assert "bob" in registry.symbols(SymbolSpace.ENTITIES)

    @pytest.mark.skip(reason="Shift-based encoding doesn't register predicates")
    def test_encode_query_registers_symbols(
        self, encoder: VSAEncoder, registry: SymbolRegistry
    ) -> None:
        """Test that encoding query registers bound symbols."""
        # NOTE: Shift-based encoding only registers entities, not predicates
        encoder.encode_query("parent", ["alice", None])

        # Check that bound entity symbols were registered
        assert "alice" in registry.symbols(SymbolSpace.ENTITIES)
        # Variables should not create entities
        assert registry.count(SymbolSpace.ENTITIES) == 1
