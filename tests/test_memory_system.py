import pytest

from metabase_agent.config.settings import Settings
from metabase_agent.memory.manager import MemoryManager
from metabase_agent.memory.models import MemoryRecord, MemoryStatus, MemoryType
from metabase_agent.memory.namespaces import user_namespace
from metabase_agent.memory.repository import InMemoryMemoryRepository
from metabase_agent.memory.vector import (
    HashEmbeddingProvider,
    InMemoryVectorIndex,
    SiliconFlowEmbeddingProvider,
)


def test_memory_manager_loads_profile_and_related_memory() -> None:
    tenant_id = "t1"
    user_id = "u1"
    repo = InMemoryMemoryRepository()
    vector = InMemoryVectorIndex()
    embeddings = HashEmbeddingProvider()
    manager = MemoryManager(repo, vector, embeddings)

    semantic_ns = user_namespace(tenant_id, user_id, MemoryType.SEMANTIC)
    profile = MemoryRecord(
        id="profile-language",
        tenant_id=tenant_id,
        user_id=user_id,
        namespace=semantic_ns,
        key="profile.language",
        memory_type=MemoryType.SEMANTIC,
        content="用户偏好中文回答。",
        value="zh-CN",
    )
    note = MemoryRecord(
        id="semantic-note",
        tenant_id=tenant_id,
        user_id=user_id,
        namespace=semantic_ns,
        key="note.orders",
        memory_type=MemoryType.SEMANTIC,
        content="用户经常分析 orders 表的 count 趋势。",
    )
    repo.put(profile)
    repo.put(note)
    vector.upsert(note, embeddings.embed(note.content))

    context = manager.load_context(tenant_id=tenant_id, user_id=user_id, query="orders count 趋势")

    assert "用户偏好中文回答" in context.rendered
    assert "orders" in context.rendered


def test_memory_manager_records_semantic_and_episodic_candidates() -> None:
    repo = InMemoryMemoryRepository()
    vector = InMemoryVectorIndex()
    manager = MemoryManager(repo, vector, HashEmbeddingProvider())

    records = manager.record_interaction(
        tenant_id="t1",
        user_id="u1",
        question="以后默认用中文回答，并且解释要直接",
        answer="好的。",
        query_result={"status": "completed"},
        query_plan={"database_name": "BigQuery-GA", "schema_name": "business_data", "table_name": "orders"},
    )

    keys = {record.key for record in records}
    assert "profile.language" in keys
    assert "profile.answer_style" in keys
    assert "profile.default_database" in keys
    assert "profile.default_schema" in keys
    assert "analytics.table_context" in keys
    assert any(key.startswith("event.analysis.") for key in keys)


def test_memory_manager_lists_and_updates_status() -> None:
    repo = InMemoryMemoryRepository()
    manager = MemoryManager(repo, InMemoryVectorIndex(), HashEmbeddingProvider())

    record = manager.put_memory(
        tenant_id="t1",
        user_id="u1",
        memory_type=MemoryType.PROCEDURAL,
        key="rule.review_sql",
        content="执行 SQL 前必须先确认。",
        status=MemoryStatus.PENDING_REVIEW,
    )

    assert record is not None
    assert manager.list_memories(tenant_id="t1", user_id="u1", memory_type=MemoryType.PROCEDURAL, status=MemoryStatus.PENDING_REVIEW) == [record]

    updated = manager.update_status(tenant_id="t1", user_id="u1", record_id=record.id, status=MemoryStatus.ACTIVE)

    assert updated is not None
    assert updated.status == MemoryStatus.ACTIVE
    assert manager.list_memories(tenant_id="t1", user_id="u1", memory_type=MemoryType.PROCEDURAL, status=MemoryStatus.ACTIVE) == [updated]


def test_siliconflow_embedding_provider_uses_configured_api(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, object], timeout: float) -> Response:
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr("metabase_agent.memory.vector.httpx.post", fake_post)
    provider = SiliconFlowEmbeddingProvider(
        Settings(
            SILICONFLOW_API_KEY="test-key",
            AGENT_EMBEDDING_MODEL="BAAI/bge-m3",
            AGENT_EMBEDDING_DIMENSIONS=3,
            OPENAI_TIMEOUT=12,
        )
    )

    embedding = provider.embed("Hello, world!")

    assert embedding == [0.1, 0.2, 0.3]
    assert captured["url"] == "https://api.siliconflow.cn/v1/embeddings"
    assert captured["headers"] == {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
    assert captured["json"] == {"input": "Hello, world!", "model": "BAAI/bge-m3"}
    assert captured["timeout"] == 12


def test_siliconflow_embedding_provider_rejects_dimension_mismatch(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    monkeypatch.setattr("metabase_agent.memory.vector.httpx.post", lambda *args, **kwargs: Response())
    provider = SiliconFlowEmbeddingProvider(
        Settings(
            SILICONFLOW_API_KEY="test-key",
            AGENT_EMBEDDING_MODEL="BAAI/bge-m3",
            AGENT_EMBEDDING_DIMENSIONS=1536,
        )
    )

    # A silent mismatch would be rejected by the pgvector table on every
    # upsert and quietly disable semantic recall — it must fail loudly.
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        provider.embed("Hello, world!")
