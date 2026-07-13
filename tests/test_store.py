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


class _FakeCollection:
    """Minimal Mongo collection double that records ops and mimics the few
    semantics MongoSessionStore relies on ($push/$slice, find_one_and_delete)."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.calls: list[tuple] = []

    def create_index(self, keys) -> None:
        self.calls.append(("create_index", keys))

    def find_one(self, flt):
        return self.docs.get(flt["_id"])

    def find_one_and_update(self, flt, update, *, upsert, return_document):
        self.calls.append(("find_one_and_update", flt, update))
        doc = self.docs.setdefault(flt["_id"], {"_id": flt["_id"], "messages": []})
        push = update["$push"]["messages"]
        doc["messages"] = (doc.get("messages", []) + push["$each"])[push["$slice"] :]
        doc.update(update["$set"])
        return doc

    def find_one_and_delete(self, flt):
        self.calls.append(("find_one_and_delete", flt))
        return self.docs.pop(flt["_id"], None)

    def replace_one(self, flt, doc, *, upsert) -> None:
        self.docs[flt["_id"]] = doc

    def delete_one(self, flt) -> None:
        self.docs.pop(flt["_id"], None)

    def delete_many(self, flt) -> None:
        cutoff = flt["updated"]["$lt"]
        for key in [k for k, v in self.docs.items() if v.get("updated", 0) < cutoff]:
            del self.docs[key]


class _FakeAdmin:
    def command(self, _name):
        return {"ok": 1}


class _FakeDB:
    def __init__(self) -> None:
        self._collections: dict[str, _FakeCollection] = {}

    def __getitem__(self, name):
        return self._collections.setdefault(name, _FakeCollection())


class _FakeMongoClient:
    def __init__(self) -> None:
        self._db = _FakeDB()
        self.admin = _FakeAdmin()

    def __getitem__(self, _database):
        return self._db


def _mongo_store():
    from metabase_agent.api.store import MongoSessionStore

    return MongoSessionStore("mongodb://x", database="db", client=_FakeMongoClient())


def test_mongo_store_message_roundtrip_and_trim() -> None:
    store = _mongo_store()
    for index in range(25):
        snapshot = store.append_message("s1", "user", f"m{index}", max_messages=20)
    assert len(snapshot) == 20
    assert snapshot[0]["content"] == "m5"
    assert store.history("s1")[-1]["content"] == "m24"
    assert store.history("missing") == []


def test_mongo_store_claim_approval_is_take_once() -> None:
    store = _mongo_store()
    store.set_approval("s1", {"sql": "SELECT 1", "mode": "tools"})
    assert store.get_approval("s1") == {"sql": "SELECT 1", "mode": "tools"}

    first = store.claim_approval("s1")
    second = store.claim_approval("s1")

    assert first == {"sql": "SELECT 1", "mode": "tools"}
    assert second is None


def test_mongo_store_table_context_and_purge() -> None:
    store = _mongo_store()
    store.set_table_context("s1", {"database_name": "BigQuery-GA"})
    assert store.get_table_context("s1") == {"database_name": "BigQuery-GA"}
    store.pop_table_context("s1")
    assert store.get_table_context("s1") is None


def test_active_store_selects_mongodb_backend(monkeypatch) -> None:
    import metabase_agent.api.app as app_module
    from metabase_agent.config.settings import get_settings

    captured: dict[str, object] = {}

    class _StubMongoStore:
        def __init__(self, uri, *, database):
            captured["uri"] = uri
            captured["database"] = database

        def purge_expired(self, ttl):
            captured["purged"] = ttl

    monkeypatch.setenv("AGENT_STORE", "mongodb")
    monkeypatch.setenv("AGENT_STORE_MONGODB_URI", "mongodb://127.0.0.1:27017")
    monkeypatch.setenv("AGENT_STORE_MONGODB_DATABASE", "sess_db")
    get_settings.cache_clear()
    app_module._SQLITE_STORE.clear()
    monkeypatch.setattr(app_module, "MongoSessionStore", _StubMongoStore)

    store = app_module._active_store()

    assert isinstance(store, _StubMongoStore)
    assert captured["uri"] == "mongodb://127.0.0.1:27017"
    assert captured["database"] == "sess_db"
    get_settings.cache_clear()
