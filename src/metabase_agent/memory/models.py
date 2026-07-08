from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class MemoryType(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


@dataclass(slots=True)
class MemoryRecord:
    id: str
    tenant_id: str
    user_id: str
    namespace: tuple[str, ...]
    key: str
    memory_type: MemoryType
    content: str
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    status: MemoryStatus = MemoryStatus.ACTIVE
    source: str = "agent"
    created_at: str = ""
    updated_at: str = ""
    last_seen: str = ""

    def __post_init__(self) -> None:
        now = utc_now_iso()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.last_seen:
            self.last_seen = self.updated_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "namespace": list(self.namespace),
            "key": self.key,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "value": self.value,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "status": self.status.value,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryRecord:
        return cls(
            id=str(payload["id"]),
            tenant_id=str(payload.get("tenant_id") or ""),
            user_id=str(payload.get("user_id") or ""),
            namespace=tuple(str(item) for item in payload.get("namespace", [])),
            key=str(payload.get("key") or ""),
            memory_type=MemoryType(str(payload.get("memory_type") or MemoryType.SEMANTIC.value)),
            content=str(payload.get("content") or ""),
            value=payload.get("value"),
            metadata=dict(payload.get("metadata") or {}),
            confidence=float(payload.get("confidence", 1.0)),
            status=MemoryStatus(str(payload.get("status") or MemoryStatus.ACTIVE.value)),
            source=str(payload.get("source") or "agent"),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            last_seen=str(payload.get("last_seen") or ""),
        )


@dataclass(slots=True)
class MemoryContext:
    profile: list[MemoryRecord] = field(default_factory=list)
    active_rules: list[MemoryRecord] = field(default_factory=list)
    related: list[MemoryRecord] = field(default_factory=list)
    rendered: str = ""

    def is_empty(self) -> bool:
        return not self.profile and not self.active_rules and not self.related and not self.rendered.strip()


@dataclass(slots=True)
class CandidateMemory:
    memory_type: MemoryType
    content: str
    key: str | None = None
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    status: MemoryStatus = MemoryStatus.ACTIVE
    source: str = "conversation"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
