import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from metabase_agent.api.app import (
    _CHECKPOINTER_CACHE,
    _GRAPH_CACHE,
    _MEMORY_MANAGER_CACHE,
    _SKILL_REGISTRY_CACHE,
    CONVERSATION_MEMORY,
    PENDING_SQL_APPROVALS,
    PENDING_TABLE_CONTEXT,
    _remember,
    create_app,
)
from metabase_agent.config.settings import get_settings


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("AGENT_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.delenv("AGENT_API_TOKEN", raising=False)
    monkeypatch.setattr("metabase_agent.api.app.MEMORY_LOADED", False)
    monkeypatch.setattr("metabase_agent.api.app.STATE_LOADED", False)
    CONVERSATION_MEMORY.clear()
    PENDING_SQL_APPROVALS.clear()
    PENDING_TABLE_CONTEXT.clear()
    _CHECKPOINTER_CACHE.clear()
    _GRAPH_CACHE.clear()
    _MEMORY_MANAGER_CACHE.clear()
    _SKILL_REGISTRY_CACHE.clear()
    yield
    get_settings.cache_clear()


def test_pending_approval_survives_app_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    first = TestClient(create_app())
    review = first.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "persist-approval"})
    assert review.json()["query_result"]["status"] == "requires_approval"

    # Simulate a process restart: drop in-memory state and force a reload from disk.
    PENDING_SQL_APPROVALS.clear()
    PENDING_TABLE_CONTEXT.clear()
    monkeypatch.setattr("metabase_agent.api.app.STATE_LOADED", False)

    second = TestClient(create_app())
    approved = second.post("/api/ask", json={"question": "确认执行", "dry_run": True, "session_id": "persist-approval"})

    data = approved.json()
    assert data["program"] == {"type": "native_sql", "database_id": 19, "sql": "SELECT 1 AS ok"}
    assert data["query_result"]["dry_run"] is True


def test_ask_api_requires_token_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_API_TOKEN", "secret")
    client = TestClient(create_app())
    body = {"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "auth-ask"}

    missing = client.post("/api/ask", json=body)
    wrong = client.post("/api/ask", json=body, headers={"X-Agent-Token": "nope"})
    ok = client.post("/api/ask", json=body, headers={"X-Agent-Token": "secret"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert ok.status_code == 200


def test_ask_api_allows_request_when_token_unset() -> None:
    client = TestClient(create_app())

    response = client.post("/api/ask", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "auth-off"})

    assert response.status_code == 200


def test_sessions_api_requires_token_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_API_TOKEN", "secret")
    client = TestClient(create_app())

    assert client.get("/api/sessions/x").status_code == 401
    assert client.get("/api/sessions/x", headers={"X-Agent-Token": "secret"}).status_code == 200


def test_ask_stream_requires_token_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_API_TOKEN", "secret")
    client = TestClient(create_app())

    response = client.post("/api/ask/stream", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "auth-stream"})

    assert response.status_code == 401


def test_memory_writes_are_atomic_for_concurrent_readers(tmp_path) -> None:
    # AGENT_MEMORY_PATH is pointed at tmp_path/memory.json by the autouse fixture.
    memory_file = tmp_path / "memory.json"
    _remember("seed", "user", "hello")
    stop = threading.Event()
    errors: list[Exception] = []

    def writer() -> None:
        counter = 0
        while not stop.is_set():
            _remember("seed", "user", "x" * 80000 + str(counter))  # large => multi-syscall flush
            counter += 1

    def reader() -> None:
        while not stop.is_set():
            try:
                text = memory_file.read_text(encoding="utf-8")
                if text:
                    json.loads(text)  # torn (half-written) read -> JSONDecodeError
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(5)] + [threading.Thread(target=reader) for _ in range(5)]
    for thread in threads:
        thread.start()
    time.sleep(0.6)
    stop.set()
    for thread in threads:
        thread.join()

    # A concurrent reader must always see a complete file (old or new), never a torn write.
    assert errors == []
    # No temp file should be left behind by the atomic write.
    assert list(tmp_path.glob("*.tmp")) == []


def test_graph_is_built_once_and_reused_across_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    from metabase_agent.agent.graph import build_graph as real_build_graph

    count = {"n": 0}

    def counting_build(settings: object) -> object:
        count["n"] += 1
        return real_build_graph(settings)

    monkeypatch.setattr("metabase_agent.api.app.build_graph", counting_build)
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "cache-a"})
    client.post("/api/ask", json={"question": "有哪些数据库？", "dry_run": True, "session_id": "cache-a"})

    assert count["n"] == 1


def test_mongodb_checkpointing_passes_session_id_as_thread_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeGraph:
        def invoke(self, payload: dict[str, object], config: dict[str, object] | None = None) -> dict[str, object]:
            seen["payload"] = payload
            seen["config"] = config
            return {
                "answer": "ok",
                "query_plan": {"intent": "test"},
                "program": None,
                "query_result": {"status": "completed"},
                "trace": [],
            }

    monkeypatch.setenv("AGENT_CHECKPOINT_BACKEND", "mongodb")
    monkeypatch.setenv("AGENT_MONGODB_URI", "mongodb://127.0.0.1:27017")
    get_settings.cache_clear()
    monkeypatch.setattr("metabase_agent.api.app._checkpointer", lambda settings: object())
    monkeypatch.setattr("metabase_agent.api.app.build_graph", lambda settings, checkpointer=None: FakeGraph())
    client = TestClient(create_app())

    response = client.post("/api/ask", json={"question": "x", "dry_run": True, "session_id": "thread-1"})

    assert response.status_code == 200
    assert seen["config"] == {"configurable": {"thread_id": "thread-1"}}


def test_home_page_loads() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "Metabase Agent" in response.text


def test_ask_api_dry_run() -> None:
    client = TestClient(create_app())

    response = client.post("/api/ask", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "dry-run-test"})

    assert response.status_code == 200
    data = response.json()
    assert "Total Revenue" in data["answer"]
    assert data["program"]["source"] == {"type": "metric", "id": 10}
    assert "trace" in data


def test_ask_api_remembers_previous_question() -> None:
    client = TestClient(create_app())

    first = client.post("/api/ask", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "memory-test"})
    second = client.post("/api/ask", json={"question": "我上次问的内容是什么？", "dry_run": True, "session_id": "memory-test"})

    assert first.status_code == 200
    assert second.status_code == 200
    data = second.json()
    assert data["answer"] == "你上次问的是：上周收入趋势怎么样？"
    assert data["query_result"] == {"status": "completed", "source": "memory"}
    assert data["session_id"] == "memory-test"


def test_ask_api_memory_is_session_scoped() -> None:
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "memory-a"})
    response = client.post("/api/ask", json={"question": "我上次问的问题是什么？", "dry_run": True, "session_id": "memory-b"})

    assert response.status_code == 200
    assert response.json()["answer"] == "我还没有记住上一轮问题。"


def test_session_api_returns_memory_for_continuous_chat() -> None:
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "session-history"})
    response = client.get("/api/sessions/session-history")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session-history"
    assert data["memory"][0] == {"role": "user", "content": "上周收入趋势怎么样？"}
    assert data["memory"][1]["role"] == "assistant"


def test_session_api_returns_pending_approval_for_reload() -> None:
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "pending-reload"})
    response = client.get("/api/sessions/pending-reload")

    assert response.status_code == 200
    assert response.json()["pending_approval"] == {"status": "requires_approval", "sql": "SELECT 1 AS ok"}


def test_memory_admin_api_writes_lists_and_activates_procedural_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    from metabase_agent.memory.manager import MemoryManager
    from metabase_agent.memory.repository import InMemoryMemoryRepository
    from metabase_agent.memory.vector import HashEmbeddingProvider, InMemoryVectorIndex

    manager = MemoryManager(InMemoryMemoryRepository(), InMemoryVectorIndex(), HashEmbeddingProvider())
    monkeypatch.setenv("AGENT_LONG_TERM_MEMORY_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr("metabase_agent.api.app._memory_manager", lambda settings: manager)
    client = TestClient(create_app())

    created = client.post(
        "/api/memories",
        json={
            "tenant_id": "t1",
            "user_id": "u1",
            "memory_type": "procedural",
            "key": "rule.sql.require_review",
            "content": "执行 SQL 前必须先让用户确认。",
            "status": "pending_review",
        },
    )

    assert created.status_code == 200
    record = created.json()["memory"]
    assert record["status"] == "pending_review"

    pending = client.get(
        "/api/memories",
        params={"tenant_id": "t1", "user_id": "u1", "memory_type": "procedural", "status": "pending_review"},
    )
    assert pending.status_code == 200
    assert pending.json()["memories"][0]["key"] == "rule.sql.require_review"

    activated = client.post(
        f"/api/memories/{record['id']}/status",
        json={"tenant_id": "t1", "user_id": "u1", "status": "active"},
    )
    assert activated.status_code == 200
    assert activated.json()["memory"]["status"] == "active"
    assert "执行 SQL 前必须先让用户确认" in manager.load_context(tenant_id="t1", user_id="u1", query="SQL 可以执行吗").rendered


def test_memory_admin_api_requires_long_term_memory_enabled() -> None:
    client = TestClient(create_app())

    response = client.get("/api/memories", params={"tenant_id": "t1", "user_id": "u1"})

    assert response.status_code == 400
    assert response.json()["detail"] == "AGENT_LONG_TERM_MEMORY_ENABLED is false"


def test_session_memory_survives_app_recreate(monkeypatch: pytest.MonkeyPatch) -> None:
    first_client = TestClient(create_app())
    first_client.post("/api/ask", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "persisted-session"})
    CONVERSATION_MEMORY.clear()
    monkeypatch.setattr("metabase_agent.api.app.MEMORY_LOADED", False)

    second_client = TestClient(create_app())
    response = second_client.get("/api/sessions/persisted-session")

    assert response.status_code == 200
    assert response.json()["memory"][0] == {"role": "user", "content": "上周收入趋势怎么样？"}


def test_config_api_returns_default_dry_run() -> None:
    client = TestClient(create_app())

    response = client.get("/api/config")

    assert response.status_code == 200
    assert "default_dry_run" in response.json()


def test_ask_api_returns_error_payload_instead_of_500(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingGraph:
        def invoke(self, _payload: dict[str, object]) -> dict[str, object]:
            raise RuntimeError("network failed")

    monkeypatch.setattr("metabase_agent.api.app.build_graph", lambda settings: FailingGraph())
    client = TestClient(create_app())

    response = client.post("/api/ask", json={"question": "真实查询", "dry_run": False})

    assert response.status_code == 200
    data = response.json()
    # Raw exception text may leak internal URLs/hosts; the API only exposes the class name.
    assert data["query_result"] == {"status": "failed", "error": "RuntimeError"}
    assert data["trace"] == [{"step": "api.ask", "status": "failed", "error": "RuntimeError"}]
    assert "查询失败" in data["answer"]
    assert "network failed" not in data["answer"]


def test_ask_api_stores_pending_sql_and_executes_after_approval() -> None:
    client = TestClient(create_app())

    review = client.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "approval-test"})
    approved = client.post("/api/ask", json={"question": "确认执行", "dry_run": True, "session_id": "approval-test"})

    assert review.status_code == 200
    review_data = review.json()
    assert review_data["query_result"]["status"] == "requires_approval"
    assert review_data["query_result"]["sql"] == "SELECT 1 AS ok"
    assert approved.status_code == 200
    approved_data = approved.json()
    assert approved_data["query_result"]["dry_run"] is True
    assert approved_data["program"] == {"type": "native_sql", "database_id": 19, "sql": "SELECT 1 AS ok"}
    assert "approval-test" not in PENDING_SQL_APPROVALS


def test_ask_api_new_sql_paste_is_not_swallowed_by_pending_approval() -> None:
    client = TestClient(create_app())

    review = client.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "hijack-test"})
    # New message carries a different SQL AND an approval keyword ("可以执行").
    # It must be treated as a NEW request, not as approval of the pending SELECT 1.
    second = client.post("/api/ask", json={"question": "SELECT 2 AS two 可以执行", "dry_run": True, "session_id": "hijack-test"})

    assert review.json()["query_result"]["sql"] == "SELECT 1 AS ok"
    data = second.json()
    assert data["query_result"]["status"] == "requires_approval"
    assert data["query_result"]["sql"] == "SELECT 2 AS two"


def test_ask_api_explicit_decision_approve_runs_pending_sql() -> None:
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "decision-approve"})
    approved = client.post("/api/ask", json={"question": "ok", "decision": "approve", "dry_run": True, "session_id": "decision-approve"})

    data = approved.json()
    assert data["program"] == {"type": "native_sql", "database_id": 19, "sql": "SELECT 1 AS ok"}
    assert data["query_result"]["dry_run"] is True
    assert "decision-approve" not in PENDING_SQL_APPROVALS


def test_ask_api_explicit_decision_reject_does_not_run_sql() -> None:
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "decision-reject"})
    rejected = client.post("/api/ask", json={"question": "no", "decision": "reject", "dry_run": True, "session_id": "decision-reject"})

    data = rejected.json()
    assert data["query_result"] == {"status": "rejected", "sql": "SELECT 1 AS ok"}
    assert "decision-reject" not in PENDING_SQL_APPROVALS


def test_ask_api_rejects_pending_sql_without_execution() -> None:
    client = TestClient(create_app())

    review = client.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "reject-test"})
    rejected = client.post("/api/ask", json={"question": "拒绝执行", "dry_run": True, "session_id": "reject-test"})

    assert review.status_code == 200
    assert review.json()["query_result"]["status"] == "requires_approval"
    assert rejected.status_code == 200
    data = rejected.json()
    assert data["answer"] == "已拒绝执行，SQL 未运行。"
    assert data["query_result"] == {"status": "rejected", "sql": "SELECT 1 AS ok"}
    assert "reject-test" not in PENDING_SQL_APPROVALS


def test_ask_stream_returns_status_and_final_events() -> None:
    client = TestClient(create_app())

    with client.stream("POST", "/api/ask/stream", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "stream-test"}) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "event: status" in body
    assert "event: final" in body
    assert "Total Revenue" in body


def test_ask_api_fills_table_follow_up_from_pending_context() -> None:
    client = TestClient(create_app())
    PENDING_TABLE_CONTEXT["follow-up-test"] = {"schema_name": "business_data", "table_name": "orders"}

    response = client.post("/api/ask", json={"question": "最近7天的每天的数据count", "dry_run": True, "session_id": "follow-up-test"})

    assert response.status_code == 200
    data = response.json()
    assert data["query_plan"]["table_name"] == "orders"
    assert data["query_plan"]["relative_days"] == 7
    assert data["query_plan"]["time_grain"] == "day"
    assert data["query_result"]["status"] == "requires_approval"


def test_ask_api_uses_previous_successful_table_for_follow_up() -> None:
    client = TestClient(create_app())

    first = client.post("/api/ask", json={"question": "business_data 下orders 最近7天的每天的数据count", "dry_run": True, "session_id": "success-follow-up"})
    first_approved = client.post("/api/ask", json={"question": "确认执行", "dry_run": True, "session_id": "success-follow-up"})
    second = client.post("/api/ask", json={"question": "最近7天的每天的数据count", "dry_run": True, "session_id": "success-follow-up"})

    assert first.status_code == 200
    assert first.json()["query_result"]["status"] == "requires_approval"
    assert first_approved.status_code == 200
    assert first_approved.json()["query_result"]["row_count"] == 2
    assert second.status_code == 200
    data = second.json()
    assert data["query_plan"]["table_name"] == "orders"
    assert data["query_plan"]["schema_name"] == "business_data"
    assert data["query_result"]["status"] == "requires_approval"


def test_ask_api_uses_previous_table_and_metric_for_growth_follow_up() -> None:
    client = TestClient(create_app())

    first = client.post("/api/ask", json={"question": "business_data 下orders 最近7天的每天的数据count", "dry_run": True, "session_id": "growth-follow-up"})
    first_approved = client.post("/api/ask", json={"question": "确认执行", "dry_run": True, "session_id": "growth-follow-up"})
    second = client.post("/api/ask", json={"question": "分析一下最近2天数量是否增长，以及哪部分有了增长", "dry_run": True, "session_id": "growth-follow-up"})

    assert first.status_code == 200
    assert first_approved.status_code == 200
    assert second.status_code == 200
    data = second.json()
    assert data["query_plan"]["table_name"] == "orders"
    assert data["query_plan"]["schema_name"] == "business_data"
    assert data["query_plan"]["aggregation_function"] == "count"
    assert data["query_plan"]["relative_days"] == 2
    assert data["query_plan"]["time_grain"] == "day"
    assert data["query_result"]["status"] == "requires_approval"


def test_ask_stream_emits_node_status_events() -> None:
    client = TestClient(create_app())

    response = client.post("/api/ask/stream", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "stream-nodes"})

    body = response.text
    assert "event: final" in body
    assert '"node": "parse"' in body
    assert '"node": "answer"' in body


def _offline_tools(settings):
    from metabase_agent.agent.tools import AgentTools as _AgentTools

    return _AgentTools(settings, dry_run=True)


def test_tools_mode_runs_loop_and_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    from metabase_agent.agent.tool_loop import ToolCall

    monkeypatch.setenv("AGENT_MODE", "tools")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()

    class _Transport:
        def __init__(self) -> None:
            self._script = [[ToolCall(id="c1", name="list_databases", arguments={})], "当前有 3 个数据库。"]

        def complete(self, messages, tools):
            return self._script.pop(0)

    monkeypatch.setattr("metabase_agent.api.app.build_tool_transport", lambda settings: _Transport())
    monkeypatch.setattr("metabase_agent.api.app.AgentTools", lambda settings, dry_run: _offline_tools(settings))
    client = TestClient(create_app())

    response = client.post("/api/ask", json={"question": "有哪些数据库？", "dry_run": False, "session_id": "tools-1"})

    body = response.json()
    assert body["answer"] == "当前有 3 个数据库。"
    assert body["query_result"]["status"] == "completed"


def test_tools_mode_falls_back_to_pipeline_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODE", "tools")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.post("/api/ask", json={"question": "当前有几个数据库？", "dry_run": True, "session_id": "tools-2"})

    assert response.json()["query_result"]["status"] == "completed"


def test_tools_mode_stream_uses_tool_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from metabase_agent.agent.tool_loop import ToolCall

    monkeypatch.setenv("AGENT_MODE", "tools")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()

    class _Transport:
        def complete(self, messages, tools):
            if any(m.get("role") == "tool" for m in messages):
                return "工具循环回答：3 行。"
            return [ToolCall(id="c1", name="run_aggregation", arguments={"database_name": "BigQuery-GA", "schema_name": "business_data", "table_name": "orders", "aggregation": "count"})]

    monkeypatch.setattr("metabase_agent.api.app.build_tool_transport", lambda settings: _Transport())
    monkeypatch.setattr("metabase_agent.api.app.AgentTools", lambda settings, dry_run: _offline_tools(settings))
    client = TestClient(create_app())

    response = client.post("/api/ask/stream", json={"question": "orders 多少行", "dry_run": False, "session_id": "tools-stream"})

    body = response.text
    assert "工具循环回答" in body
    assert '"rows": [[3]]' in body


def test_dry_run_forces_pipeline_even_in_tools_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODE", "tools")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()

    def _boom(settings):
        raise AssertionError("tools transport must not be built during dry-run")

    monkeypatch.setattr("metabase_agent.api.app.build_tool_transport", _boom)
    client = TestClient(create_app())

    response = client.post("/api/ask", json={"question": "当前有几个数据库？", "dry_run": True, "session_id": "dry-tools"})

    assert response.json()["query_result"]["status"] == "completed"


def test_require_token_rejects_when_no_token_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_REQUIRE_TOKEN", "true")
    monkeypatch.setenv("AGENT_API_TOKEN", "")
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.post("/api/ask", json={"question": "x", "dry_run": True, "session_id": "rt"})

    assert response.status_code == 401


def test_require_token_allows_with_correct_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_REQUIRE_TOKEN", "true")
    monkeypatch.setenv("AGENT_API_TOKEN", "secret")
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.post("/api/ask", json={"question": "当前有几个数据库？", "dry_run": True, "session_id": "rt2"}, headers={"X-Agent-Token": "secret"})

    assert response.status_code == 200


def test_sqlite_store_shares_state_across_app_instances(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "state.db")
    monkeypatch.setenv("AGENT_STORE", "sqlite")
    monkeypatch.setenv("AGENT_STATE_PATH", db)
    get_settings.cache_clear()
    import metabase_agent.api.app as app_module

    app_module._SQLITE_STORE.clear()

    worker_a = TestClient(create_app())
    worker_b = TestClient(create_app())

    worker_a.post("/api/ask", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "shared"})
    second = worker_b.post("/api/ask", json={"question": "我上次问的内容是什么？", "dry_run": True, "session_id": "shared"})

    assert "上周收入趋势怎么样" in second.json()["answer"]


def test_sqlite_session_ttl_evicts_stale_sessions(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    db = str(tmp_path / "state.db")
    monkeypatch.setenv("AGENT_STORE", "sqlite")
    monkeypatch.setenv("AGENT_STATE_PATH", db)
    monkeypatch.setenv("AGENT_SESSION_TTL_SECONDS", "0.05")
    get_settings.cache_clear()
    import metabase_agent.api.app as app_module

    app_module._SQLITE_STORE.clear()
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "ttl"})
    time.sleep(0.1)
    second = client.post("/api/ask", json={"question": "我上次问的内容是什么？", "dry_run": True, "session_id": "ttl"})

    assert second.json()["answer"] == "我还没有记住上一轮问题。"


def test_ask_api_negated_execute_phrase_is_rejection_not_approval() -> None:
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "negation-test"})
    # "please don't execute" mentions execution but is a rejection; the old
    # approval-first phrase matching executed the pending SQL here.
    second = client.post("/api/ask", json={"question": "please don't execute", "dry_run": True, "session_id": "negation-test"})

    data = second.json()
    assert data["query_result"]["status"] == "rejected"
    assert "negation-test" not in PENDING_SQL_APPROVALS


def test_ask_api_chinese_negated_execute_phrase_is_rejection() -> None:
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "negation-zh"})
    second = client.post("/api/ask", json={"question": "先不要执行", "dry_run": True, "session_id": "negation-zh"})

    assert second.json()["query_result"]["status"] == "rejected"


def test_ask_api_second_approve_finds_nothing_to_execute() -> None:
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "double-approve"})
    first = client.post("/api/ask", json={"question": "确认执行", "dry_run": True, "session_id": "double-approve"})
    second = client.post("/api/ask", json={"question": "确认执行", "dry_run": True, "session_id": "double-approve"})

    # The approval is take-once: a duplicate approve (double-click, concurrent
    # request) must not execute the same SQL a second time.
    assert first.json()["query_result"]["dry_run"] is True
    assert first.json()["program"] == {"type": "native_sql", "database_id": 19, "sql": "SELECT 1 AS ok"}
    second_data = second.json()
    assert second_data["program"] != {"type": "native_sql", "database_id": 19, "sql": "SELECT 1 AS ok"}
    assert second_data["query_result"].get("sql") != "SELECT 1 AS ok"


def test_ask_api_approval_bound_to_reviewed_program() -> None:
    client = TestClient(create_app())

    review = client.post("/api/ask", json={"question": "请执行 SELECT 1 AS ok", "dry_run": True, "session_id": "binding-test"})
    assert review.json()["query_result"]["status"] == "requires_approval"

    # Simulate drift between review and approval (e.g. non-deterministic
    # re-planning): the stored fingerprint no longer matches what would run.
    PENDING_SQL_APPROVALS["binding-test"]["program_hash"] = "tampered"

    approved = client.post("/api/ask", json={"question": "确认执行", "dry_run": True, "session_id": "binding-test"})

    data = approved.json()
    assert data["query_result"]["status"] == "requires_approval"
    assert "不一致" in data["answer"]


def test_ask_api_database_answer_resumes_pending_clarification() -> None:
    client = TestClient(create_app())

    first = client.post("/api/ask", json={"question": "orders 这个表最近7天的每天的数据count，并分析是否增长", "dry_run": True, "session_id": "clarify-db"})
    # The reply is ONLY a database name — it must resume the pending
    # aggregation, not be parsed as a brand-new "list tables" question.
    second = client.post("/api/ask", json={"question": "BigQuery-GA", "dry_run": True, "session_id": "clarify-db"})

    first_data = first.json()
    assert first_data["query_result"]["status"] == "requires_clarification"
    assert first_data["query_result"]["clarification_type"] == "database"
    second_data = second.json()
    assert second_data["query_plan"]["intent"] == "table_aggregation"
    assert second_data["query_plan"]["table_name"] == "orders"
    assert second_data["query_plan"]["relative_days"] == 7
    assert second_data["query_plan"]["time_grain"] == "day"
    assert second_data["query_result"]["status"] == "requires_approval"


def test_ask_api_database_answer_with_chatty_suffix_still_resumes() -> None:
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "orders 这个表最近7天的每天的数据count", "dry_run": True, "session_id": "clarify-db-2"})
    second = client.post("/api/ask", json={"question": "BigQuery-GA 中的，你在上文的输出中没有获取到吗", "dry_run": True, "session_id": "clarify-db-2"})

    data = second.json()
    assert data["query_plan"]["intent"] == "table_aggregation"
    assert data["query_plan"]["table_name"] == "orders"
    assert data["query_result"]["status"] == "requires_approval"


def test_ask_api_new_question_mentioning_database_is_not_hijacked() -> None:
    client = TestClient(create_app())

    client.post("/api/ask", json={"question": "orders 这个表最近7天的每天的数据count", "dry_run": True, "session_id": "clarify-db-3"})
    # This mentions the database but is clearly its own request — it must NOT
    # be rewritten back into the pending orders aggregation.
    second = client.post("/api/ask", json={"question": "BigQuery-GA 下有哪些表", "dry_run": True, "session_id": "clarify-db-3"})

    data = second.json()
    assert data["query_plan"].get("table_name") != "orders"
    assert data["query_plan"]["intent"] != "table_aggregation"


def test_ask_api_reuses_last_database_from_session_context() -> None:
    client = TestClient(create_app())

    first = client.post("/api/ask", json={"question": "BigQuery-GA 这个数据库下有哪些表", "dry_run": True, "session_id": "default-db"})
    # No database in this question — the session remembered BigQuery-GA.
    second = client.post("/api/ask", json={"question": "orders 这个表最近7天的每天的数据count", "dry_run": True, "session_id": "default-db"})

    assert first.json()["query_result"]["status"] == "completed"
    data = second.json()
    assert data["query_result"]["status"] == "requires_approval"
    assert data["query_plan"]["database_name"] == "BigQuery-GA"
