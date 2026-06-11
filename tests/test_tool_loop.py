from __future__ import annotations

from typing import Any

from metabase_agent.agent.tool_loop import ToolCall, run_tool_loop
from metabase_agent.agent.tools import AgentTools, tool_schemas
from metabase_agent.config.settings import Settings


class _ScriptedTransport:
    def __init__(self, script: list[Any]) -> None:
        self._script = script
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._script.pop(0)


def _tools() -> AgentTools:
    return AgentTools(Settings(AGENT_DRY_RUN=True), dry_run=True)


def test_tool_schemas_cover_core_capabilities() -> None:
    names = {schema["name"] for schema in tool_schemas()}

    assert names == {"list_databases", "list_tables", "list_fields", "run_aggregation", "run_sql"}


def test_list_databases_tool_returns_sample_names() -> None:
    result = _tools().dispatch("list_databases", {})

    assert "BigQuery-GA" in result["databases"]


def test_run_aggregation_tool_is_policy_checked_and_executes_in_dry_run() -> None:
    result = _tools().dispatch(
        "run_aggregation",
        {"database_name": "BigQuery-GA", "schema_name": "business_data", "table_name": "orders", "aggregation": "count"},
    )

    assert result["status"] == "completed"
    assert result["row_count"] == 1


def test_run_sql_tool_blocks_mutations() -> None:
    result = _tools().dispatch("run_sql", {"sql": "DROP TABLE users"})

    assert result["status"] == "blocked"


def test_loop_returns_answer_after_tool_call() -> None:
    transport = _ScriptedTransport(
        [
            [ToolCall(id="c1", name="list_databases", arguments={})],
            "当前有 3 个数据库。",
        ]
    )

    outcome = run_tool_loop("有哪些数据库？", [], transport, _tools())

    assert outcome.answer == "当前有 3 个数据库。"
    assert outcome.status == "completed"
    assert any(event.get("tool") == "list_databases" for event in outcome.trace)


def test_loop_suspends_for_sql_approval() -> None:
    transport = _ScriptedTransport(
        [
            [ToolCall(id="c1", name="run_sql", arguments={"sql": "SELECT 1 AS ok"})],
        ]
    )

    outcome = run_tool_loop("跑一下 SELECT 1", [], transport, _tools())

    assert outcome.status == "requires_approval"
    assert outcome.pending_sql == "SELECT 1 AS ok"
    assert outcome.messages, "message history must be captured for resume"


def test_loop_resume_executes_approved_sql() -> None:
    transport = _ScriptedTransport(["已执行，返回 1 行。"])
    tools = _tools()

    suspended_messages = [
        {"role": "user", "content": "跑一下 SELECT 1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "run_sql", "arguments": {"sql": "SELECT 1 AS ok"}}]},
    ]
    outcome = run_tool_loop("", suspended_messages, transport, tools, approved_sql="SELECT 1 AS ok", approved_tool_call_id="c1")

    assert outcome.status == "completed"
    assert outcome.answer == "已执行，返回 1 行。"


def test_loop_stops_after_max_iterations() -> None:
    transport = _ScriptedTransport([[ToolCall(id=f"c{i}", name="list_databases", arguments={})] for i in range(10)])

    outcome = run_tool_loop("循环", [], transport, _tools(), max_iterations=3)

    assert outcome.status == "exhausted"
    assert len([c for c in transport.calls]) == 3


def test_responses_adapter_parses_function_call(monkeypatch) -> None:
    from metabase_agent.semantics import llm_client

    captured: dict[str, Any] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "output": [
                    {"type": "function_call", "name": "list_databases", "arguments": "{}", "call_id": "fc_1"}
                ]
            }

    def _fake_post(url: str, **kwargs: Any) -> _Resp:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(llm_client.httpx, "post", _fake_post)
    transport = llm_client.ResponsesToolTransport(Settings(OPENAI_API_KEY="k", OPENAI_WIRE_API="responses"))

    reply = transport.complete([{"role": "user", "content": "dbs?"}], tool_schemas())

    assert isinstance(reply, list)
    assert reply[0].name == "list_databases"
    assert captured["url"].endswith("/responses")
    assert captured["json"]["tools"][0]["type"] == "function"


def test_responses_adapter_parses_final_text(monkeypatch) -> None:
    from metabase_agent.semantics import llm_client

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "有 3 个库。"}]}]}

    monkeypatch.setattr(llm_client.httpx, "post", lambda url, **kwargs: _Resp())
    transport = llm_client.ResponsesToolTransport(Settings(OPENAI_API_KEY="k", OPENAI_WIRE_API="responses"))

    reply = transport.complete([{"role": "user", "content": "dbs?"}], tool_schemas())

    assert reply == "有 3 个库。"


def test_chat_adapter_parses_tool_calls(monkeypatch) -> None:
    from metabase_agent.semantics import llm_client

    class _Fn:
        name = "list_databases"
        arguments = "{}"

    class _TC:
        id = "call_1"
        function = _Fn()

    class _Msg:
        content = None
        tool_calls = [_TC()]

    class _Choice:
        message = _Msg()

    class _Completion:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs: Any) -> _Completion:
            return _Completion()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

        chat = _Chat()

    monkeypatch.setattr(llm_client, "OpenAI", _FakeOpenAI)
    transport = llm_client.ChatToolTransport(Settings(OPENAI_API_KEY="k", OPENAI_WIRE_API="chat_completions"))

    reply = transport.complete([{"role": "user", "content": "dbs?"}], tool_schemas())

    assert isinstance(reply, list)
    assert reply[0].name == "list_databases"


def test_loop_captures_last_tool_result_for_ui() -> None:
    transport = _ScriptedTransport(
        [
            [ToolCall(id="c1", name="run_aggregation", arguments={"database_name": "BigQuery-GA", "schema_name": "business_data", "table_name": "orders", "aggregation": "count"})],
            "orders 共 3 行。",
        ]
    )

    outcome = run_tool_loop("orders 多少行", [], transport, _tools())

    assert outcome.status == "completed"
    assert outcome.last_result is not None
    assert outcome.last_result["data"]["rows"] == [[3]]


def test_run_sql_targets_requested_database_id() -> None:
    captured: dict[str, Any] = {}

    class _StubClient:
        def list_databases(self) -> Any:
            return {"data": [{"id": 7, "name": "warehouse"}, {"id": 19, "name": "BigQuery-GA"}]}

        def execute_native_query(self, database_id: int, sql: str) -> dict[str, Any]:
            captured["database_id"] = database_id
            return {"status": "completed", "row_count": 0, "data": {"cols": [], "rows": []}}

    tools = AgentTools(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="k"), dry_run=False)
    tools._client = _StubClient()  # type: ignore[assignment]

    result = tools.dispatch("run_sql", {"sql": "SELECT 1", "database_name": "warehouse"})

    assert result["status"] == "completed"
    assert captured["database_id"] == 7


def test_run_sql_defaults_to_bigquery_database_id() -> None:
    captured: dict[str, Any] = {}

    class _StubClient:
        def list_databases(self) -> Any:
            return {"data": [{"id": 19, "name": "BigQuery-GA"}]}

        def execute_native_query(self, database_id: int, sql: str) -> dict[str, Any]:
            captured["database_id"] = database_id
            return {"status": "completed", "row_count": 0, "data": {"cols": [], "rows": []}}

    tools = AgentTools(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="k", METABASE_BIGQUERY_DATABASE_ID=19), dry_run=False)
    tools._client = _StubClient()  # type: ignore[assignment]

    tools.dispatch("run_sql", {"sql": "SELECT 1"})

    assert captured["database_id"] == 19
