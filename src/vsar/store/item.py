"""Item schema for VSAR knowledge base."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import jax.numpy as jnp


class ItemKind(Enum):
    """Types of items in the knowledge base."""
    FACT = "fact"
    RULE = "rule"
    AXIOM = "axiom"
    EDGE = "edge"      # Argumentation edge
    CASE = "case"      # Abductive case
    MAP = "map"        # Analogy mapping


@dataclass
class Provenance:
    """
    Provenance metadata for tracking item origins.

    Attributes:
        source: Where this item came from (e.g., "user", "inference", "file:data.vsar")
        timestamp: When this item was created
        agent: Optional agent identifier for epistemic contexts
        trace: Derivation trace showing how this item was derived
    """
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent: Optional[str] = None
    trace: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """String representation."""
        agent_str = f", agent={self.agent}" if self.agent else ""
        trace_str = f", trace={len(self.trace)} steps" if self.trace else ""
        return f"Provenance(source={self.source}{agent_str}{trace_str})"


@dataclass
class Item:
    """
    A knowledge base item with metadata.

    Items represent facts, rules, axioms, or other knowledge elements
    along with their hypervector encodings and associated metadata.

    Attributes:
        vec: FHRR hypervector encoding
        kind: Type of item (FACT, RULE, etc.)
        weight: Probability, confidence, or belief mass (0-1)
        priority: Priority for defeasible reasoning (higher = more specific)
        agent: Agent identifier for epistemic reasoning
        provenance: Source and derivation metadata
        tags: Additional tags for categorization

    Example:
        >>> vec = encoder.encode_atom(Atom("parent", [Constant("alice"), Constant("bob")]))
        >>> item = Item(
        ...     vec=vec,
        ...     kind=ItemKind.FACT,
        ...     weight=1.0,
        ...     provenance=Provenance(source="user")
        ... )
    """
    vec: jnp.ndarray
    kind: ItemKind
    weight: float = 1.0
    priority: Optional[float] = None
    agent: Optional[str] = None
    provenance: Provenance = field(default_factory=lambda: Provenance(source="unknown"))
    tags: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Validate item after initialization."""
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"Weight must be in [0, 1], got {self.weight}")

        if self.priority is not None and self.priority < 0:
            raise ValueError(f"Priority must be non-negative, got {self.priority}")

    def is_fact(self) -> bool:
        """Check if this is a fact item."""
        return self.kind == ItemKind.FACT

    def is_rule(self) -> bool:
        """Check if this is a rule item."""
        return self.kind == ItemKind.RULE

    def with_weight(self, weight: float) -> 'Item':
        """Create a copy with updated weight."""
        return Item(
            vec=self.vec,
            kind=self.kind,
            weight=weight,
            priority=self.priority,
            agent=self.agent,
            provenance=self.provenance,
            tags=self.tags
        )

    def with_provenance(self, provenance: Provenance) -> 'Item':
        """Create a copy with updated provenance."""
        return Item(
            vec=self.vec,
            kind=self.kind,
            weight=self.weight,
            priority=self.priority,
            agent=self.agent,
            provenance=provenance,
            tags=self.tags
        )

    def __repr__(self) -> str:
        """String representation."""
        weight_str = f", weight={self.weight:.2f}" if self.weight != 1.0 else ""
        priority_str = f", priority={self.priority}" if self.priority is not None else ""
        agent_str = f", agent={self.agent}" if self.agent else ""
        tags_str = f", tags={self.tags}" if self.tags else ""
        return f"Item({self.kind.value}{weight_str}{priority_str}{agent_str}{tags_str})"
