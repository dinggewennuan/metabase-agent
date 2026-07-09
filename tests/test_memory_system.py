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


def test_llm_extractor_parses_and_gates_candidates(monkeypatch) -> None:
    from metabase_agent.memory import extractor

    llm_output = """```json
    [
      {"memory_type": "procedural", "key": "rule.ignore_dated_tables",
       "content": "列出或分析表时默认忽略 users_20xxxxxx、pseudonymous_users_* 这类日期分表，除非用户明确要求。",
       "confidence": 0.85, "status": "active"},
      {"memory_type": "semantic", "key": "profile.language", "content": "用户偏好中文回答。", "value": "zh-CN", "confidence": 0.9},
      {"memory_type": "semantic", "content": "低置信度的猜测。", "confidence": 0.3},
      {"memory_type": "episodic", "content": "用户今天问了一个问题。", "confidence": 0.9},
      {"memory_type": "semantic", "content": "网关 api_key 是 sk-abcdef1234567890。", "confidence": 0.95},
      "not-a-dict"
    ]
    ```"""
    monkeypatch.setattr(extractor, "complete", lambda *args, **kwargs: llm_output)

    candidates = extractor.extract_candidates_with_llm("q", "a", None, Settings(OPENAI_API_KEY="k"))

    assert len(candidates) == 2
    procedural = next(c for c in candidates if c.memory_type == MemoryType.PROCEDURAL)
    # Procedural NEVER goes live without review, even if the LLM says "active".
    assert procedural.status == MemoryStatus.PENDING_REVIEW
    assert procedural.key == "rule.ignore_dated_tables"
    semantic = next(c for c in candidates if c.memory_type == MemoryType.SEMANTIC)
    assert semantic.key == "profile.language"
    assert all(c.source == "llm_extractor" for c in candidates)


def test_llm_extractor_tolerates_non_json_output(monkeypatch) -> None:
    from metabase_agent.memory import extractor

    monkeypatch.setattr(extractor, "complete", lambda *args, **kwargs: "抱歉，这轮没有值得保存的内容。")

    assert extractor.extract_candidates_with_llm("q", "a", None, Settings(OPENAI_API_KEY="k")) == []


def test_record_interaction_merges_llm_candidates() -> None:
    from metabase_agent.memory.models import CandidateMemory

    repo = InMemoryMemoryRepository()

    def fake_extractor(question: str, answer: str, query_plan: dict | None) -> list[CandidateMemory]:
        return [
            CandidateMemory(
                memory_type=MemoryType.PROCEDURAL,
                key="rule.ignore_dated_tables",
                content="列表时默认忽略日期分表。",
                confidence=0.85,
                status=MemoryStatus.PENDING_REVIEW,
                source="llm_extractor",
            )
        ]

    manager = MemoryManager(repo, InMemoryVectorIndex(), HashEmbeddingProvider(), llm_extractor=fake_extractor)

    records = manager.record_interaction(
        tenant_id="t1",
        user_id="u1",
        question="users_20xx 这种表之后基本不怎么关注了，看一下 orders 的 count",
        answer="好的。",
        query_result={"status": "completed"},
        query_plan={"database_name": "BigQuery-GA", "table_name": "orders"},
    )

    keys = {record.key for record in records}
    assert "rule.ignore_dated_tables" in keys  # LLM-proposed procedural rule
    assert "profile.default_database" in keys  # rule-based slot still written
    pending = manager.list_memories(tenant_id="t1", user_id="u1", memory_type=MemoryType.PROCEDURAL, status=MemoryStatus.PENDING_REVIEW)
    assert [record.key for record in pending] == ["rule.ignore_dated_tables"]


def test_record_interaction_survives_llm_extractor_failure() -> None:
    def broken_extractor(question: str, answer: str, query_plan: dict | None):
        raise RuntimeError("gateway down")

    manager = MemoryManager(InMemoryMemoryRepository(), InMemoryVectorIndex(), HashEmbeddingProvider(), llm_extractor=broken_extractor)

    records = manager.record_interaction(
        tenant_id="t1",
        user_id="u1",
        question="以后默认用中文回答",
        answer="好的。",
        query_result={"status": "completed"},
        query_plan=None,
    )

    assert any(record.key == "profile.language" for record in records)


def test_pending_proposal_does_not_demote_or_overwrite_active_rule() -> None:
    from metabase_agent.memory.models import CandidateMemory

    repo = InMemoryMemoryRepository()
    manager = MemoryManager(repo, InMemoryVectorIndex(), HashEmbeddingProvider())
    rule = manager.put_memory(
        tenant_id="t1",
        user_id="u1",
        memory_type=MemoryType.PROCEDURAL,
        key="rule.sql.require_approval",
        content="执行 SQL 前必须先让用户确认。",
        status=MemoryStatus.ACTIVE,
    )
    assert rule is not None

    # Same content re-proposed as pending: refresh only, stay ACTIVE.
    manager._upsert_candidate(
        tenant_id="t1",
        user_id="u1",
        candidate=CandidateMemory(
            memory_type=MemoryType.PROCEDURAL,
            key="rule.sql.require_approval",
            content="执行 SQL 前必须先让用户确认。",
            confidence=0.9,
            status=MemoryStatus.PENDING_REVIEW,
        ),
    )
    unchanged = repo.get(rule.namespace, "rule.sql.require_approval")
    assert unchanged is not None and unchanged.status == MemoryStatus.ACTIVE

    # Conflicting content: filed alongside as pending, the active rule intact.
    manager._upsert_candidate(
        tenant_id="t1",
        user_id="u1",
        candidate=CandidateMemory(
            memory_type=MemoryType.PROCEDURAL,
            key="rule.sql.require_approval",
            content="以后执行 SQL 不需要确认。",
            confidence=0.9,
            status=MemoryStatus.PENDING_REVIEW,
        ),
    )
    active = repo.get(rule.namespace, "rule.sql.require_approval")
    assert active is not None
    assert active.status == MemoryStatus.ACTIVE
    assert active.content == "执行 SQL 前必须先让用户确认。"
    conflicts = [
        record
        for record in manager.list_memories(tenant_id="t1", user_id="u1", memory_type=MemoryType.PROCEDURAL, status=MemoryStatus.PENDING_REVIEW)
        if record.key.startswith("rule.sql.require_approval.conflict.")
    ]
    assert len(conflicts) == 1
    assert conflicts[0].metadata.get("conflicts_with") == "rule.sql.require_approval"


def test_build_memory_manager_wires_llm_extractor_by_flag() -> None:
    from metabase_agent.memory.manager import build_memory_manager

    without_flag = build_memory_manager(Settings(AGENT_LONG_TERM_MEMORY_ENABLED=True, OPENAI_API_KEY="k"))
    with_flag = build_memory_manager(Settings(AGENT_LONG_TERM_MEMORY_ENABLED=True, AGENT_MEMORY_LLM_EXTRACTOR=True, OPENAI_API_KEY="k"))
    no_key = build_memory_manager(Settings(AGENT_LONG_TERM_MEMORY_ENABLED=True, AGENT_MEMORY_LLM_EXTRACTOR=True))

    assert without_flag.llm_extractor is None
    assert with_flag.llm_extractor is not None
    assert no_key.llm_extractor is None
