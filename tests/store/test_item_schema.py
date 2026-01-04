"""Tests for item schema and metadata (Phase 3.1)."""

import pytest
from datetime import datetime
import jax.numpy as jnp

from vsar.store.item import Item, ItemKind, Provenance
from vsar.store.belief import BeliefState, Literal


class TestItemSchema:
    """Test Item dataclass and metadata."""

    def test_create_fact_item(self):
        """Create Item with kind=FACT."""
        vec = jnp.ones(512, dtype=jnp.complex64)
        prov = Provenance(source="user")

        item = Item(
            vec=vec,
            kind=ItemKind.FACT,
            weight=0.9,
            provenance=prov
        )

        assert item.is_fact()
        assert not item.is_rule()
        assert item.weight == 0.9
        assert item.provenance.source == "user"
        assert item.priority is None
        assert item.agent is None

    def test_create_rule_item(self):
        """Create Item with kind=RULE, priority."""
        vec = jnp.ones(512, dtype=jnp.complex64)
        prov = Provenance(source="inference", trace=["step1", "step2"])

        item = Item(
            vec=vec,
            kind=ItemKind.RULE,
            weight=1.0,
            priority=10.0,
            provenance=prov
        )

        assert not item.is_fact()
        assert item.is_rule()
        assert item.priority == 10.0
        assert len(item.provenance.trace) == 2

    def test_item_with_agent(self):
        """Create Item with agent for epistemic reasoning."""
        vec = jnp.ones(512, dtype=jnp.complex64)

        item = Item(
            vec=vec,
            kind=ItemKind.FACT,
            agent="alice",
            provenance=Provenance(source="kb_alice")
        )

        assert item.agent == "alice"

    def test_item_with_tags(self):
        """Create Item with tags."""
        vec = jnp.ones(512, dtype=jnp.complex64)

        item = Item(
            vec=vec,
            kind=ItemKind.FACT,
            tags={"medical", "diagnostic"}
        )

        assert "medical" in item.tags
        assert "diagnostic" in item.tags
        assert len(item.tags) == 2

    def test_item_weight_validation(self):
        """Test that weight must be in [0, 1]."""
        vec = jnp.ones(512, dtype=jnp.complex64)

        # Valid weights
        item1 = Item(vec=vec, kind=ItemKind.FACT, weight=0.0)
        item2 = Item(vec=vec, kind=ItemKind.FACT, weight=1.0)
        item3 = Item(vec=vec, kind=ItemKind.FACT, weight=0.5)

        # Invalid weights
        with pytest.raises(ValueError):
            Item(vec=vec, kind=ItemKind.FACT, weight=-0.1)

        with pytest.raises(ValueError):
            Item(vec=vec, kind=ItemKind.FACT, weight=1.5)

    def test_item_priority_validation(self):
        """Test that priority must be non-negative."""
        vec = jnp.ones(512, dtype=jnp.complex64)

        # Valid priorities
        item1 = Item(vec=vec, kind=ItemKind.RULE, priority=0.0)
        item2 = Item(vec=vec, kind=ItemKind.RULE, priority=100.0)

        # Invalid priority
        with pytest.raises(ValueError):
            Item(vec=vec, kind=ItemKind.RULE, priority=-1.0)

    def test_item_with_weight(self):
        """Test creating copy with updated weight."""
        vec = jnp.ones(512, dtype=jnp.complex64)
        item = Item(vec=vec, kind=ItemKind.FACT, weight=0.5)

        new_item = item.with_weight(0.8)

        assert new_item.weight == 0.8
        assert item.weight == 0.5  # Original unchanged
        assert new_item.kind == item.kind

    def test_item_with_provenance(self):
        """Test creating copy with updated provenance."""
        vec = jnp.ones(512, dtype=jnp.complex64)
        prov1 = Provenance(source="user")
        item = Item(vec=vec, kind=ItemKind.FACT, provenance=prov1)

        prov2 = Provenance(source="inference", trace=["derived"])
        new_item = item.with_provenance(prov2)

        assert new_item.provenance.source == "inference"
        assert item.provenance.source == "user"  # Original unchanged

    def test_item_kinds(self):
        """Test all item kinds."""
        vec = jnp.ones(512, dtype=jnp.complex64)

        fact = Item(vec=vec, kind=ItemKind.FACT)
        rule = Item(vec=vec, kind=ItemKind.RULE)
        axiom = Item(vec=vec, kind=ItemKind.AXIOM)
        edge = Item(vec=vec, kind=ItemKind.EDGE)

        assert fact.kind.value == "fact"
        assert rule.kind.value == "rule"
        assert axiom.kind.value == "axiom"
        assert edge.kind.value == "edge"


class TestProvenance:
    """Test Provenance dataclass."""

    def test_provenance_defaults(self):
        """Test provenance with defaults."""
        prov = Provenance(source="user")

        assert prov.source == "user"
        assert isinstance(prov.timestamp, datetime)
        assert prov.agent is None
        assert prov.trace == []

    def test_provenance_with_trace(self):
        """Test provenance with derivation trace."""
        prov = Provenance(
            source="inference",
            agent="reasoner",
            trace=["rule1", "rule2", "rule3"]
        )

        assert prov.source == "inference"
        assert prov.agent == "reasoner"
        assert len(prov.trace) == 3
        assert prov.trace[0] == "rule1"

    def test_provenance_repr(self):
        """Test provenance string representation."""
        prov = Provenance(source="test", agent="alice", trace=["step1"])

        repr_str = repr(prov)
        assert "test" in repr_str
        assert "alice" in repr_str


class TestBeliefState:
    """Test paraconsistent belief tracking."""

    def test_create_belief_state(self):
        """Create BeliefState with supp_pos and supp_neg."""
        belief = BeliefState(supp_pos=0.7, supp_neg=0.0)

        assert belief.supp_pos == 0.7
        assert belief.supp_neg == 0.0

    def test_belief_consistent(self):
        """Test consistent belief (only positive support)."""
        belief = BeliefState(supp_pos=0.9, supp_neg=0.0)

        assert belief.is_consistent()
        assert not belief.is_contradictory()
        assert not belief.is_unknown()

    def test_belief_contradictory(self):
        """Test contradictory belief (both positive and negative support)."""
        belief = BeliefState(supp_pos=0.7, supp_neg=0.5)

        assert not belief.is_consistent()
        assert belief.is_contradictory()
        assert not belief.is_unknown()

    def test_belief_unknown(self):
        """Test unknown belief (no support)."""
        belief = BeliefState(supp_pos=0.0, supp_neg=0.0)

        assert not belief.is_consistent()
        assert not belief.is_contradictory()
        assert belief.is_unknown()

    def test_belief_validation(self):
        """Test that support values must be in [0, 1]."""
        # Valid
        belief1 = BeliefState(supp_pos=0.0, supp_neg=0.0)
        belief2 = BeliefState(supp_pos=1.0, supp_neg=1.0)

        # Invalid
        with pytest.raises(ValueError):
            BeliefState(supp_pos=-0.1, supp_neg=0.0)

        with pytest.raises(ValueError):
            BeliefState(supp_pos=0.0, supp_neg=1.5)

    def test_update_positive_support(self):
        """Test adding support for L."""
        belief = BeliefState(supp_pos=0.5, supp_neg=0.0)

        updated = belief.update_positive(0.3)

        assert updated.supp_pos == 0.8
        assert updated.supp_neg == 0.0
        assert belief.supp_pos == 0.5  # Original unchanged

    def test_update_negative_support(self):
        """Test adding support for ¬L."""
        belief = BeliefState(supp_pos=0.0, supp_neg=0.3)

        updated = belief.update_negative(0.4)

        assert updated.supp_pos == 0.0
        assert updated.supp_neg == 0.7

    def test_update_support_capped_at_one(self):
        """Test that support is capped at 1.0."""
        belief = BeliefState(supp_pos=0.8, supp_neg=0.0)

        updated = belief.update_positive(0.5)

        assert updated.supp_pos == 1.0  # Capped

    def test_net_support(self):
        """Test net support calculation."""
        belief1 = BeliefState(supp_pos=0.8, supp_neg=0.2)
        assert belief1.net_support() == pytest.approx(0.6)

        belief2 = BeliefState(supp_pos=0.3, supp_neg=0.7)
        assert belief2.net_support() == pytest.approx(-0.4)

    def test_belief_repr(self):
        """Test belief state string representation."""
        belief = BeliefState(supp_pos=0.9, supp_neg=0.0)

        repr_str = repr(belief)
        assert "0.90" in repr_str
        assert "CONSISTENT" in repr_str


class TestLiteral:
    """Test Literal dataclass."""

    def test_create_literal(self):
        """Create a literal."""
        lit = Literal("parent", ("alice", "bob"))

        assert lit.predicate == "parent"
        assert lit.args == ("alice", "bob")
        assert not lit.negated

    def test_create_negated_literal(self):
        """Create a negated literal."""
        lit = Literal("parent", ("alice", "bob"), negated=True)

        assert lit.negated

    def test_negate_literal(self):
        """Test negation of literal."""
        lit = Literal("parent", ("alice", "bob"))
        neg_lit = lit.negate()

        assert not lit.negated
        assert neg_lit.negated
        assert neg_lit.predicate == "parent"
        assert neg_lit.args == ("alice", "bob")

    def test_literal_to_key(self):
        """Test conversion to string key."""
        lit1 = Literal("parent", ("alice", "bob"))
        lit2 = Literal("parent", ("alice", "bob"), negated=True)

        key1 = lit1.to_key()
        key2 = lit2.to_key()

        assert key1 == "parent(alice,bob)"
        assert key2 == "~parent(alice,bob)"
        assert key1 != key2

    def test_literal_equality(self):
        """Test literal equality."""
        lit1 = Literal("parent", ("alice", "bob"))
        lit2 = Literal("parent", ("alice", "bob"))
        lit3 = Literal("parent", ("alice", "carol"))

        assert lit1 == lit2
        assert lit1 != lit3

    def test_literal_hash(self):
        """Test that literals can be used in sets/dicts."""
        lit1 = Literal("parent", ("alice", "bob"))
        lit2 = Literal("parent", ("alice", "bob"))

        lit_set = {lit1, lit2}
        assert len(lit_set) == 1  # Same literal

    def test_literal_repr(self):
        """Test literal string representation."""
        lit1 = Literal("parent", ("alice", "bob"))
        lit2 = Literal("parent", ("alice", "bob"), negated=True)

        assert str(lit1) == "parent(alice, bob)"
        assert str(lit2) == "¬parent(alice, bob)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
