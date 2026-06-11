"""SQLite-backed session store for multi-worker deployments.

Selected via AGENT_STORE=sqlite. Each call opens a short-lived connection so the
store is safe to share across processes/workers; WAL mode allows concurrent reads
while a writer holds the lock.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, messages TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS approvals (session_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS table_context (session_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated REAL NOT NULL);
"""


class SqliteStore:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def history(self, session_id: str) -> list[dict[str, str]]:
        with self._connect() as conn:
            row = conn.execute("SELECT messages FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(row[0]) if row else []

    def append_message(self, session_id: str, role: str, content: str, max_messages: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT messages FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            messages: list[dict[str, str]] = json.loads(row[0]) if row else []
            messages.append({"role": role, "content": content})
            messages = messages[-max_messages:]
            conn.execute(
                "INSERT INTO sessions(session_id, messages, updated) VALUES(?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET messages=excluded.messages, updated=excluded.updated",
                (session_id, json.dumps(messages, ensure_ascii=False), time.time()),
            )
            conn.execute("COMMIT")
        return messages

    def get_approval(self, session_id: str) -> dict[str, Any] | None:
        return self._get("approvals", session_id)

    def set_approval(self, session_id: str, data: dict[str, Any]) -> None:
        self._set("approvals", session_id, data)

    def pop_approval(self, session_id: str) -> None:
        self._pop("approvals", session_id)

    def get_table_context(self, session_id: str) -> dict[str, Any] | None:
        return self._get("table_context", session_id)

    def set_table_context(self, session_id: str, data: dict[str, Any]) -> None:
        self._set("table_context", session_id, data)

    def pop_table_context(self, session_id: str) -> None:
        self._pop("table_context", session_id)

    def purge_expired(self, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            return
        cutoff = time.time() - ttl_seconds
        with self._connect() as conn:
            for table in ("sessions", "approvals", "table_context"):
                conn.execute(f"DELETE FROM {table} WHERE updated < ?", (cutoff,))

    def _get(self, table: str, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(f"SELECT data FROM {table} WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def _set(self, table: str, session_id: str, data: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {table}(session_id, data, updated) VALUES(?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET data=excluded.data, updated=excluded.updated",
                (session_id, json.dumps(data, ensure_ascii=False), time.time()),
            )

    def _pop(self, table: str, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(f"DELETE FROM {table} WHERE session_id=?", (session_id,))
