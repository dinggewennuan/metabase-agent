from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
from openai import OpenAI

from metabase_agent.config.settings import Settings
from metabase_agent.memory.models import MemoryRecord

_LOGGER = logging.getLogger("metabase_agent")


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
        return _check_dimensions(list(response.data[0].embedding), self._dimensions)


class SiliconFlowEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.siliconflow_api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is required for SiliconFlow embeddings")
        self._api_key = settings.siliconflow_api_key
        self._base_url = settings.siliconflow_base_url.rstrip("/")
        self._model = settings.agent_embedding_model
        self._timeout = settings.openai_timeout
        self._dimensions = settings.agent_embedding_dimensions
        # Flips to False once the model rejects the MRL dimensions param, so
        # fixed-dimension models (bge-m3) don't pay a 400 on every embed.
        self._send_dimensions = True

    def embed(self, text: str) -> list[float]:
        body: dict[str, Any] = {"input": text, "model": self._model}
        if self._dimensions > 0 and self._send_dimensions:
            # MRL models (Qwen3-Embedding family, ~4096 native dims) must be
            # asked for the configured dimension: pgvector's HNSW index caps
            # at 2000 dims, so the native size can never be stored as-is.
            body["dimensions"] = self._dimensions
        try:
            response = self._post_embeddings(body)
        except httpx.HTTPStatusError:
            if "dimensions" not in body:
                raise
            # Fixed-dimension models (e.g. BAAI/bge-m3) reject the dimensions
            # param — retry without it and remember, so every later embed goes
            # straight through instead of eating a 400 first. The output is
            # still length-checked against the configured dimension.
            response = self._post_embeddings({"input": text, "model": self._model})
            self._send_dimensions = False
            _LOGGER.info("siliconflow model %s rejects the dimensions param; omitting it from now on", self._model)
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("empty SiliconFlow embedding response")
        first = data[0]
        if not isinstance(first, dict) or not isinstance(first.get("embedding"), list):
            raise RuntimeError("invalid SiliconFlow embedding response")
        return _check_dimensions([float(value) for value in first["embedding"]], self._dimensions)

    def _post_embeddings(self, body: dict[str, Any]) -> httpx.Response:
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                # Never send httpx's default "python-httpx/x.y" UA: it is a
                # stock WAF block rule (the same one that 403'd the LLM gateway).
                "User-Agent": "metabase-agent/0.1",
            },
            json=body,
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            # The provider's error body names the actual cause (invalid key /
            # real-name verification / model permission / region block) —
            # without it a 403 is undebuggable from the status line alone.
            _LOGGER.warning("siliconflow embeddings HTTP %s: %s", response.status_code, response.text[:300])
        response.raise_for_status()
        return response


class VectorIndex(Protocol):
    def upsert(self, record: MemoryRecord, embedding: Sequence[float]) -> None: ...

    def search(self, tenant_id: str, user_id: str, query_embedding: Sequence[float], *, memory_types: Sequence[str], limit: int = 5) -> list[str]: ...


class NullVectorIndex:
    def ping(self) -> None:
        return None

    def upsert(self, record: MemoryRecord, embedding: Sequence[float]) -> None:
        return None

    def search(self, tenant_id: str, user_id: str, query_embedding: Sequence[float], *, memory_types: Sequence[str], limit: int = 5) -> list[str]:
        return []


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._items: dict[str, tuple[MemoryRecord, list[float]]] = {}

    def ping(self) -> None:
        return None

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


def pgvector_ddl_statements(table: str, dimensions: int) -> list[str]:
    """Idempotent DDL for the memory-embeddings table, its extension and indexes.

    Shared by the init script and the in-code auto-create path so they can
    never drift.
    """
    table = _validate_identifier(table)
    vector_index = _validate_identifier(f"{table}_vector_idx")
    filter_index = _validate_identifier(f"{table}_filter_idx")
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
          id text PRIMARY KEY,
          tenant_id text NOT NULL,
          user_id text NOT NULL,
          scope text NOT NULL,
          memory_type text NOT NULL,
          memory_id text NOT NULL,
          content text NOT NULL,
          embedding vector({int(dimensions)}) NOT NULL,
          metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
          status text NOT NULL,
          created_at timestamptz NOT NULL,
          updated_at timestamptz NOT NULL
        )
        """,
        f"CREATE INDEX IF NOT EXISTS {vector_index} ON {table} USING hnsw (embedding vector_cosine_ops)",
        f"CREATE INDEX IF NOT EXISTS {filter_index} ON {table} (tenant_id, user_id, memory_type, status)",
    ]


class PgVectorIndex:
    def __init__(self, dsn: str, *, table: str = "memory_embeddings", dimensions: int = 1536, auto_create: bool = False) -> None:
        try:
            import psycopg
            from psycopg.types.json import Json
        except ImportError as exc:  # pragma: no cover - only hit without optional dependency.
            raise RuntimeError("psycopg is required for PgVectorIndex") from exc
        self._psycopg = psycopg
        self._json = Json
        self._dsn = dsn
        self._table = _validate_identifier(table)
        self._dimensions = dimensions
        if auto_create:
            self.ensure_schema()

    def ensure_schema(self) -> None:
        """Create the database (best-effort), extension, table and indexes if missing.

        So enabling long-term memory doesn't require a separate manual
        `init_pgvector_memory.py` run. If the table exists with a DIFFERENT
        vector dimension (e.g. created at 1536 before switching to a 1024-dim
        model), it is dropped and recreated: the embeddings table is a
        rebuildable index — MongoDB remains the memory source of truth.
        """
        self._ensure_database()
        with self._psycopg.connect(self._dsn) as conn:
            existing = self._existing_dimensions(conn)
            if existing is not None and existing != self._dimensions:
                _LOGGER.warning(
                    "pgvector table %s is vector(%s) but AGENT_EMBEDDING_DIMENSIONS=%s; "
                    "dropping and recreating it (embeddings are derived data; MongoDB memory records are unaffected)",
                    self._table,
                    existing,
                    self._dimensions,
                )
                conn.execute(f"DROP TABLE {self._table}")
            for statement in pgvector_ddl_statements(self._table, self._dimensions):
                conn.execute(statement)

    def _existing_dimensions(self, conn: Any) -> int | None:
        """Declared vector dimension of the existing table, or None if absent."""
        row = conn.execute("SELECT to_regclass(%s)", (self._table,)).fetchone()
        if not row or row[0] is None:
            return None
        # For pgvector columns atttypmod IS the declared dimension.
        dim_row = conn.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
            (self._table,),
        ).fetchone()
        if not dim_row or not isinstance(dim_row[0], int) or dim_row[0] <= 0:
            return None
        return int(dim_row[0])

    def _ensure_database(self) -> None:
        """Create the target Postgres database if it does not exist.

        Postgres never auto-creates databases, so we connect to the `postgres`
        maintenance DB and issue CREATE DATABASE. Best-effort: needs CREATEDB
        privilege; on failure the caller surfaces a clear error via ping().
        """
        try:
            with self._psycopg.connect(self._dsn):
                return
        except self._psycopg.OperationalError as exc:
            if "does not exist" not in str(exc).lower():
                raise  # server unreachable / auth error — not a missing-db case
        info = self._psycopg.conninfo.conninfo_to_dict(self._dsn)
        dbname = info.get("dbname")
        if not dbname:
            return
        maintenance = self._psycopg.conninfo.make_conninfo(**{**info, "dbname": "postgres"})
        with self._psycopg.connect(maintenance, autocommit=True) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{dbname}"')

    def ping(self) -> None:
        # Verifies both connectivity AND that the table exists (missing table
        # is the most common silent long-term-memory failure).
        with self._psycopg.connect(self._dsn) as conn:
            conn.execute(f"SELECT 1 FROM {self._table} LIMIT 1")

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


def _check_dimensions(embedding: list[float], expected: int) -> list[float]:
    # Fail loudly here: a mismatched vector would otherwise be rejected by the
    # pgvector table on every upsert and silently disable semantic recall.
    if expected > 0 and len(embedding) != expected:
        raise RuntimeError(
            f"embedding dimension mismatch: provider returned {len(embedding)} dims but "
            f"AGENT_EMBEDDING_DIMENSIONS={expected}; the pgvector table must use the same dimension "
            f"(e.g. BAAI/bge-m3 needs AGENT_EMBEDDING_DIMENSIONS=1024)"
        )
    return embedding


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
