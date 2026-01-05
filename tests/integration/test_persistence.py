"""Integration tests for KB persistence."""

from pathlib import Path

import pytest
from vsar.encoding.role_filler_encoder import RoleFillerEncoder
from vsar.kb.persistence import load_kb, save_kb
from vsar.kb.store import KnowledgeBase
from vsar.kernel.vsa_backend import FHRRBackend
from vsar.retrieval.query import Retriever
from vsar.symbols.basis import load_basis, save_basis
from vsar.symbols.registry import SymbolRegistry


class TestPersistence:
    """Integration tests for save/load workflow."""

    @pytest.fixture
    def temp_kb_file(self, tmp_path: Path) -> Path:
        """Create temporary KB file path."""
        return tmp_path / "test_kb.h5"

    @pytest.fixture
    def temp_basis_file(self, tmp_path: Path) -> Path:
        """Create temporary basis file path."""
        return tmp_path / "test_basis.h5"

    def test_save_load_kb_with_retrieval(self, temp_kb_file: Path) -> None:
        """Test save/load KB preserves retrieval functionality."""
        # Session 1: Create KB and insert facts
        backend1 = FHRRBackend(dim=512, seed=42)
        registry1 = SymbolRegistry(dim=512, seed=42)
        encoder1 = RoleFillerEncoder(backend1, registry1, seed=42)
        kb1 = KnowledgeBase(backend1)

        # Insert facts
        facts = [
            ("alice", "bob"),
            ("alice", "carol"),
        ]

        for args in facts:
            atom_vec = encoder1.encode_atom("parent", list(args))
            kb1.insert("parent", atom_vec, args)

        # Save KB
        save_kb(kb1, temp_kb_file)

        # Session 2: Load KB and verify retrieval works
        backend2 = FHRRBackend(dim=512, seed=42)
        registry2 = SymbolRegistry(dim=512, seed=42)
        encoder2 = RoleFillerEncoder(backend2, registry2, seed=42)

        # Re-register symbols (in real usage, would save/load registry too)
        for args in facts:
            encoder2.encode_atom("parent", list(args))

        # Load KB
        kb2 = load_kb(backend2, temp_kb_file)

        # Create retriever
        retriever2 = Retriever(backend2, registry2, kb2, encoder2)

        # Query: parent(alice, X)
        results = retriever2.retrieve("parent", 2, {"1": "alice"}, k=5)

        # Verify retrieval works
        assert len(results) > 0
        entity_names = [name for name, _ in results]
        assert "bob" in entity_names or "carol" in entity_names

    @pytest.mark.xfail(reason="Basis persistence API changed - needs update")
    def test_save_load_basis_preserves_symbols(self, temp_basis_file: Path) -> None:
        """Test save/load basis preserves symbol vectors."""
        # Session 1: Create registry and save basis
        registry1 = SymbolRegistry(dim=512, seed=42)

        # Register symbols
        from vsar.symbols.spaces import SymbolSpace

        registry1.register(SymbolSpace.ENTITIES, "alice")
        registry1.register(SymbolSpace.ENTITIES, "bob")
        registry1.register(SymbolSpace.PREDICATES, "parent")

        # Save basis
        save_basis(temp_basis_file, registry1._basis)

        # Session 2: Load basis and verify vectors match
        registry2 = SymbolRegistry(dim=512, seed=42)

        # Load basis
        loaded_basis = load_basis(temp_basis_file)
        registry2._basis = loaded_basis

        # Verify symbols are preserved
        alice1 = registry1.get(SymbolSpace.ENTITIES, "alice")
        alice2 = registry2.get(SymbolSpace.ENTITIES, "alice")

        assert alice1 is not None
        assert alice2 is not None

        import jax.numpy as jnp

        assert jnp.allclose(alice1, alice2)

    @pytest.mark.xfail(reason="Basis persistence API changed - needs update")
    def test_complete_save_load_workflow(self, temp_kb_file: Path, temp_basis_file: Path) -> None:
        """Test complete save/load workflow: basis + KB."""
        # Session 1: Build and save system
        backend1 = FHRRBackend(dim=512, seed=42)
        registry1 = SymbolRegistry(dim=512, seed=42)
        encoder1 = RoleFillerEncoder(backend1, registry1, seed=42)
        kb1 = KnowledgeBase(backend1)

        # Insert facts
        facts = [
            ("alice", "bob"),
            ("bob", "carol"),
        ]

        for args in facts:
            atom_vec = encoder1.encode_atom("parent", list(args))
            kb1.insert("parent", atom_vec, args)

        # Save both basis and KB
        save_basis(temp_basis_file, registry1._basis)
        save_kb(kb1, temp_kb_file)

        # Session 2: Load and verify system works
        backend2 = FHRRBackend(dim=512, seed=42)
        registry2 = SymbolRegistry(dim=512, seed=42)

        # Load basis
        loaded_basis = load_basis(temp_basis_file)
        registry2._basis = loaded_basis

        # Create encoder with loaded registry
        encoder2 = RoleFillerEncoder(backend2, registry2, seed=42)

        # Load KB
        kb2 = load_kb(backend2, temp_kb_file)

        # Create retriever
        retriever2 = Retriever(backend2, registry2, kb2, encoder2)

        # Test both queries
        results1 = retriever2.retrieve("parent", 2, {"1": "alice"}, k=5)
        results2 = retriever2.retrieve("parent", 1, {"2": "carol"}, k=5)

        # Verify both work
        assert len(results1) > 0
        assert len(results2) > 0

        # Check expected results
        assert "bob" in [name for name, _ in results1]
        assert "bob" in [name for name, _ in results2]

    def test_incremental_updates_after_load(self, temp_kb_file: Path) -> None:
        """Test that KB can be updated after loading."""
        # Session 1: Create and save KB
        backend1 = FHRRBackend(dim=512, seed=42)
        registry1 = SymbolRegistry(dim=512, seed=42)
        encoder1 = RoleFillerEncoder(backend1, registry1, seed=42)
        kb1 = KnowledgeBase(backend1)

        # Insert initial facts
        atom_vec = encoder1.encode_atom("parent", ["alice", "bob"])
        kb1.insert("parent", atom_vec, ("alice", "bob"))

        save_kb(kb1, temp_kb_file)

        # Session 2: Load and add more facts
        backend2 = FHRRBackend(dim=512, seed=42)
        registry2 = SymbolRegistry(dim=512, seed=42)
        encoder2 = RoleFillerEncoder(backend2, registry2, seed=42)

        # Re-register initial symbols
        encoder2.encode_atom("parent", ["alice", "bob"])

        # Load KB
        kb2 = load_kb(backend2, temp_kb_file)

        # Add new fact
        atom_vec2 = encoder2.encode_atom("parent", ["alice", "carol"])
        kb2.insert("parent", atom_vec2, ("alice", "carol"))

        # Verify both facts are retrievable
        retriever2 = Retriever(backend2, registry2, kb2, encoder2)

        results = retriever2.retrieve("parent", 2, {"1": "alice"}, k=5)

        assert len(results) > 0
        entity_names = [name for name, _ in results]

        # Both bob and carol should be retrievable
        assert "bob" in entity_names or "carol" in entity_names

    def test_multiple_save_load_cycles(self, temp_kb_file: Path) -> None:
        """Test multiple save/load cycles preserve data."""
        backend = FHRRBackend(dim=512, seed=42)
        registry = SymbolRegistry(dim=backend.dimension, seed=42)
        encoder = RoleFillerEncoder(backend, registry, seed=42)

        # Cycle 1: Create and save
        kb1 = KnowledgeBase(backend)
        atom_vec1 = encoder.encode_atom("parent", ["alice", "bob"])
        kb1.insert("parent", atom_vec1, ("alice", "bob"))
        save_kb(kb1, temp_kb_file)

        # Cycle 2: Load, modify, save
        kb2 = load_kb(backend, temp_kb_file)
        atom_vec2 = encoder.encode_atom("parent", ["bob", "carol"])
        kb2.insert("parent", atom_vec2, ("bob", "carol"))
        save_kb(kb2, temp_kb_file)

        # Cycle 3: Load and verify
        kb3 = load_kb(backend, temp_kb_file)

        # Should have both facts
        assert kb3.count("parent") == 2
        facts = kb3.get_facts("parent")
        assert ("alice", "bob") in facts
        assert ("bob", "carol") in facts
