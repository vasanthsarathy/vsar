"""Trace event classes for VSAR execution."""

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass
class TraceEvent:
    """A single event in the VSAR execution trace.

    Events form a DAG where parent_ids reference earlier events.
    Common event types:
    - "query": Query encoding operation
    - "unbind": Unbinding operation in retrieval
    - "cleanup": Cleanup operation in retrieval
    - "retrieval": Final retrieval results

    Attributes:
        id: Unique event identifier
        timestamp: Event creation time (seconds since epoch)
        type: Event type (query, unbind, cleanup, retrieval)
        payload: Event-specific data
        parent_ids: IDs of parent events in the DAG
    """

    id: str
    type: str
    payload: dict[str, Any]
    parent_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization.

        Returns:
            Dictionary representation of the event
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "type": self.type,
            "payload": self.payload,
            "parent_ids": self.parent_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceEvent":
        """Create event from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            TraceEvent instance
        """
        return cls(
            id=data["id"],
            type=data["type"],
            payload=data["payload"],
            parent_ids=data.get("parent_ids", []),
            timestamp=data.get("timestamp", time()),
        )
