import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from metabase_agent.api.app import (
    _GRAPH_CACHE,
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
    _GRAPH_CACHE.clear()
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
    assert data["query_result"] == {"status": "failed", "error": "network failed"}
    assert data["trace"] == [{"step": "api.ask", "status": "failed", "error": "network failed"}]
    assert "查询失败" in data["answer"]


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


def test_ask_stream_emits_node_status_events() -> None:
    client = TestClient(create_app())

    response = client.post("/api/ask/stream", json={"question": "上周收入趋势怎么样？", "dry_run": True, "session_id": "stream-nodes"})

    body = response.text
    assert "event: final" in body
    assert '"node": "parse"' in body
    assert '"node": "answer"' in body


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
    client = TestClient(create_app())

    response = client.post("/api/ask", json={"question": "有哪些数据库？", "dry_run": True, "session_id": "tools-1"})

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
    client = TestClient(create_app())

    response = client.post("/api/ask/stream", json={"question": "orders 多少行", "dry_run": True, "session_id": "tools-stream"})

    body = response.text
    assert "工具循环回答" in body
    assert '"rows": [[3]]' in body
