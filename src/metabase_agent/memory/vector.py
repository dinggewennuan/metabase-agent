from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
from openai import OpenAI

from metabase_agent.config.settings import Settings
from metabase_agent.memory.models import MemoryRecord


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    """Deterministic local embedding for tests and offline development.

    It is not semantically meaningful, but it keeps the indexing path executable
    when an external embedding model is unavailable.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
        self._client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url, timeout=settings.openai_timeout)
        self._model = settings.agent_embedding_model
        self._dimensions = settings.agent_embedding_dimensions

    def embed(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {"model": self._model, "input": text}
        if self._dimensions > 0:
            kwargs["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**kwargs)
        return list(response.data[0].embedding)


class SiliconFlowEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.siliconflow_api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for SiliconFlow embeddings")
        self._api_key = settings.siliconflow_api_key
        self._base_url = settings.siliconflow_base_url.rstrip("/")
        self._model = settings.agent_embedding_model
        self._timeout = settings.openai_timeout

    def embed(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"input": text, "model": self._model},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("empty SiliconFlow embedding response")
        first = data[0]
        if not isinstance(first, dict) or not isinstance(first.get("embedding"), list):
            raise RuntimeError("invalid SiliconFlow embedding response")
        return [float(value) for value in first["embedding"]]


class VectorIndex(Protocol):
    def upsert(self, record: MemoryRecord, embedding: Sequence[float]) -> None: ...

    def search(self, tenant_id: str, user_id: str, query_embedding: Sequence[float], *, memory_types: Sequence[str], limit: int = 5) -> list[str]: ...


class NullVectorIndex:
    def upsert(self, record: MemoryRecord, embedding: Sequence[float]) -> None:
        return None

    def search(self, tenant_id: str, user_id: str, query_embedding: Sequence[float], *, memory_types: Sequence[str], limit: int = 5) -> list[str]:
        return []


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._items: dict[str, tuple[MemoryRecord, list[float]]] = {}

    def upsert(self, record: MemoryRecord, embedding: Sequence[float]) -> None:
        self._items[record.id] = (record, list(embedding))

    def search(self, tenant_id: str, user_id: str, query_embedding: Sequence[float], *, memory_types: Sequence[str], limit: int = 5) -> list[str]:
        query = list(query_embedding)
        scored: list[tuple[float, str]] = []
        allowed = set(memory_types)
        for record_id, (record, vector) in self._items.items():
            if record.tenant_id != tenant_id or record.user_id != user_id or record.memory_type.value not in allowed:
                continue
            scored.append((_cosine(query, vector), record_id))
        scored.sort(reverse=True)
        return [record_id for score, record_id in scored[:limit] if score > 0]


class PgVectorIndex:
    def __init__(self, dsn: str, *, table: str = "memory_embeddings") -> None:
        try:
            import psycopg
            from psycopg.types.json import Json
        except ImportError as exc:  # pragma: no cover - only hit without optional dependency.
            raise RuntimeError("psycopg is required for PgVectorIndex") from exc
        self._psycopg = psycopg
        self._json = Json
        self._dsn = dsn
        self._table = _validate_identifier(table)

    def upsert(self, record: MemoryRecord, embedding: Sequence[float]) -> None:
        vector = _pg_vector_literal(embedding)
        metadata: dict[str, Any] = record.metadata
        sql = f"""
        INSERT INTO {self._table}
          (id, tenant_id, user_id, scope, memory_type, memory_id, content, embedding, metadata, status, created_at, updated_at)
        VALUES
          (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
          content = EXCLUDED.content,
          embedding = EXCLUDED.embedding,
          metadata = EXCLUDED.metadata,
          status = EXCLUDED.status,
          updated_at = EXCLUDED.updated_at
        """
        with self._psycopg.connect(self._dsn) as conn:
            conn.execute(
                sql,
                (
                    record.id,
                    record.tenant_id,
                    record.user_id,
                    _scope_from_namespace(record.namespace),
                    record.memory_type.value,
                    record.id,
                    record.content,
                    vector,
                    self._json(metadata),
                    record.status.value,
                    record.created_at,
                    record.updated_at,
                ),
            )

    def search(self, tenant_id: str, user_id: str, query_embedding: Sequence[float], *, memory_types: Sequence[str], limit: int = 5) -> list[str]:
        if not memory_types:
            return []
        vector = _pg_vector_literal(query_embedding)
        placeholders = ", ".join(["%s"] * len(memory_types))
        sql = f"""
        SELECT memory_id
        FROM {self._table}
        WHERE tenant_id = %s
          AND user_id = %s
          AND status = 'active'
          AND memory_type IN ({placeholders})
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        params: list[object] = [tenant_id, user_id, *memory_types, vector, limit]
        with self._psycopg.connect(self._dsn) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [str(row[0]) for row in rows]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _pg_vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"


def _validate_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"invalid PostgreSQL identifier: {value}")
    return value


def _scope_from_namespace(namespace: tuple[str, ...]) -> str:
    if "user" in namespace:
        return "user"
    if "org" in namespace:
        return "org"
    if "project" in namespace:
        return "project"
    return "app"
