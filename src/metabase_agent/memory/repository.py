from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Protocol

from metabase_agent.memory.models import MemoryRecord, MemoryStatus, utc_now_iso


class MemoryRepository(Protocol):
    def get_by_id(self, record_id: str) -> MemoryRecord | None: ...

    def get(self, namespace: tuple[str, ...], key: str) -> MemoryRecord | None: ...

    def put(self, record: MemoryRecord) -> None: ...

    def list_namespace(self, namespace: tuple[str, ...], *, status: MemoryStatus | None = None, limit: int = 50) -> list[MemoryRecord]: ...

    def list_user(
        self,
        tenant_id: str,
        user_id: str,
        *,
        memory_type: str | None = None,
        status: MemoryStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]: ...

    def get_many(self, ids: Sequence[str]) -> list[MemoryRecord]: ...


class NullMemoryRepository:
    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        return None

    def get(self, namespace: tuple[str, ...], key: str) -> MemoryRecord | None:
        return None

    def put(self, record: MemoryRecord) -> None:
        return None

    def list_namespace(self, namespace: tuple[str, ...], *, status: MemoryStatus | None = None, limit: int = 50) -> list[MemoryRecord]:
        return []

    def list_user(
        self,
        tenant_id: str,
        user_id: str,
        *,
        memory_type: str | None = None,
        status: MemoryStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        return []

    def get_many(self, ids: Sequence[str]) -> list[MemoryRecord]:
        return []


class InMemoryMemoryRepository:
    def __init__(self, records: Iterable[MemoryRecord] | None = None) -> None:
        self._by_id: dict[str, MemoryRecord] = {}
        self._by_namespace_key: dict[tuple[tuple[str, ...], str], str] = {}
        for record in records or []:
            self.put(record)

    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        return self._by_id.get(record_id)

    def get(self, namespace: tuple[str, ...], key: str) -> MemoryRecord | None:
        record_id = self._by_namespace_key.get((namespace, key))
        return self._by_id.get(record_id) if record_id else None

    def put(self, record: MemoryRecord) -> None:
        self._by_id[record.id] = record
        self._by_namespace_key[(record.namespace, record.key)] = record.id

    def list_namespace(self, namespace: tuple[str, ...], *, status: MemoryStatus | None = None, limit: int = 50) -> list[MemoryRecord]:
        records = [
            record
            for record in self._by_id.values()
            if record.namespace == namespace and (status is None or record.status == status)
        ]
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return records[:limit]

    def list_user(
        self,
        tenant_id: str,
        user_id: str,
        *,
        memory_type: str | None = None,
        status: MemoryStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        records = [
            record
            for record in self._by_id.values()
            if record.tenant_id == tenant_id
            and record.user_id == user_id
            and (memory_type is None or record.memory_type.value == memory_type)
            and (status is None or record.status == status)
        ]
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return records[:limit]

    def get_many(self, ids: Sequence[str]) -> list[MemoryRecord]:
        return [record for record_id in ids if (record := self._by_id.get(record_id)) is not None]


class MongoMemoryRepository:
    """Mongo-backed memory repository.

    This intentionally stores normalized MemoryRecord documents. It can sit on
    top of the same MongoDB deployment used by LangGraph's MongoDBStore, but it
    keeps the application memory schema explicit and testable.
    """

    def __init__(self, uri: str, *, database: str, collection: str = "agent_memories") -> None:
        try:
            from pymongo import ASCENDING, MongoClient
        except ImportError as exc:  # pragma: no cover - only hit without optional dependency.
            raise RuntimeError("pymongo is required for MongoMemoryRepository") from exc

        self._client = MongoClient(uri)
        self._collection = self._client[database][collection]
        self._collection.create_index([("namespace", ASCENDING), ("key", ASCENDING)], unique=True)
        self._collection.create_index([("id", ASCENDING)], unique=True)
        self._collection.create_index([("tenant_id", ASCENDING), ("user_id", ASCENDING), ("memory_type", ASCENDING), ("status", ASCENDING)])

    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        payload = self._collection.find_one({"id": record_id})
        return MemoryRecord.from_dict(payload) if payload else None

    def get(self, namespace: tuple[str, ...], key: str) -> MemoryRecord | None:
        payload = self._collection.find_one({"namespace": list(namespace), "key": key})
        return MemoryRecord.from_dict(payload) if payload else None

    def put(self, record: MemoryRecord) -> None:
        payload = record.to_dict()
        payload["updated_at"] = payload.get("updated_at") or utc_now_iso()
        self._collection.replace_one({"namespace": payload["namespace"], "key": payload["key"]}, payload, upsert=True)

    def list_namespace(self, namespace: tuple[str, ...], *, status: MemoryStatus | None = None, limit: int = 50) -> list[MemoryRecord]:
        query: dict[str, object] = {"namespace": list(namespace)}
        if status is not None:
            query["status"] = status.value
        cursor = self._collection.find(query).sort("updated_at", -1).limit(limit)
        return [MemoryRecord.from_dict(payload) for payload in cursor]

    def list_user(
        self,
        tenant_id: str,
        user_id: str,
        *,
        memory_type: str | None = None,
        status: MemoryStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        query: dict[str, object] = {"tenant_id": tenant_id, "user_id": user_id}
        if memory_type is not None:
            query["memory_type"] = memory_type
        if status is not None:
            query["status"] = status.value
        cursor = self._collection.find(query).sort("updated_at", -1).limit(limit)
        return [MemoryRecord.from_dict(payload) for payload in cursor]

    def get_many(self, ids: Sequence[str]) -> list[MemoryRecord]:
        if not ids:
            return []
        by_id = {payload["id"]: MemoryRecord.from_dict(payload) for payload in self._collection.find({"id": {"$in": list(ids)}})}
        return [record for record_id in ids if (record := by_id.get(record_id)) is not None]


def group_by_namespace(records: Iterable[MemoryRecord]) -> dict[tuple[str, ...], list[MemoryRecord]]:
    grouped: dict[tuple[str, ...], list[MemoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.namespace].append(record)
    return grouped
