from __future__ import annotations

import time

from metabase_agent.api.store import SqliteStore


def _store(tmp_path) -> SqliteStore:
    return SqliteStore(str(tmp_path / "state.db"))


def test_memory_append_and_history_roundtrip(tmp_path) -> None:
    store = _store(tmp_path)

    store.append_message("s1", "user", "hello", max_messages=20)
    snapshot = store.append_message("s1", "assistant", "hi", max_messages=20)

    assert snapshot == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    assert store.history("s1") == snapshot
    assert store.history("missing") == []


def test_memory_trims_to_max_messages(tmp_path) -> None:
    store = _store(tmp_path)

    for index in range(25):
        store.append_message("s1", "user", f"m{index}", max_messages=20)

    history = store.history("s1")
    assert len(history) == 20
    assert history[0]["content"] == "m5"
    assert history[-1]["content"] == "m24"


def test_approval_set_get_pop(tmp_path) -> None:
    store = _store(tmp_path)

    store.set_approval("s1", {"sql": "SELECT 1", "mode": "tools"})
    assert store.get_approval("s1") == {"sql": "SELECT 1", "mode": "tools"}

    store.pop_approval("s1")
    assert store.get_approval("s1") is None


def test_table_context_set_get_pop(tmp_path) -> None:
    store = _store(tmp_path)

    store.set_table_context("s1", {"schema_name": "business_data", "table_name": "orders"})
    assert store.get_table_context("s1") == {"schema_name": "business_data", "table_name": "orders"}

    store.pop_table_context("s1")
    assert store.get_table_context("s1") is None


def test_state_is_shared_across_store_instances(tmp_path) -> None:
    db = str(tmp_path / "state.db")
    writer = SqliteStore(db)
    writer.set_approval("s1", {"sql": "SELECT 1"})

    reader = SqliteStore(db)
    assert reader.get_approval("s1") == {"sql": "SELECT 1"}


def test_purge_expired_removes_old_entries(tmp_path) -> None:
    store = _store(tmp_path)
    store.set_approval("old", {"sql": "SELECT 1"})
    store.append_message("old", "user", "x", max_messages=20)
    time.sleep(0.05)

    store.purge_expired(ttl_seconds=0.01)

    assert store.get_approval("old") is None
    assert store.history("old") == []


def test_purge_disabled_when_ttl_zero(tmp_path) -> None:
    store = _store(tmp_path)
    store.set_approval("s1", {"sql": "SELECT 1"})

    store.purge_expired(ttl_seconds=0)

    assert store.get_approval("s1") == {"sql": "SELECT 1"}


def test_claim_approval_is_take_once(tmp_path) -> None:
    store = _store(tmp_path)
    store.set_approval("s1", {"sql": "SELECT 1", "mode": "tools"})

    first = store.claim_approval("s1")
    second = store.claim_approval("s1")

    # Two concurrent approve requests must not both see the pending SQL.
    assert first == {"sql": "SELECT 1", "mode": "tools"}
    assert second is None
    assert store.get_approval("s1") is None
