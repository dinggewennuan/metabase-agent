from __future__ import annotations

from typing import Any

import httpx

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


def test_run_aggregation_tool_falls_back_to_dataset_when_agent_query_rejects() -> None:
    request = httpx.Request("POST", "https://example.test/api/agent/v2/query")
    response = httpx.Response(400, request=request)
    dataset_payloads: list[dict[str, Any]] = []

    class _StubClient:
        def list_databases(self) -> Any:
            return {"data": [{"id": 19, "name": "BigQuery-GA", "engine": "bigquery"}]}

        def get_database_schema(self, database_id: int, schema_name: str) -> Any:
            return [{"id": 1332, "name": "orders", "schema": "business_data"}]

        def get_table_query_metadata(self, table_id: int) -> dict[str, Any]:
            return {"fields": [{"id": 305, "name": "created_at", "display_name": "created_at", "base_type": "type/DateTime"}]}

        def query(self, program: dict[str, Any]) -> dict[str, Any]:
            raise httpx.HTTPStatusError("bad request", request=request, response=response)

        def execute_mbql_query(self, payload: dict[str, Any]) -> dict[str, Any]:
            dataset_payloads.append(payload)
            return {"status": "completed", "row_count": 1, "data": {"cols": [], "rows": [[1]]}}

    tools = AgentTools(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="k"), dry_run=False)
    tools._client = _StubClient()  # type: ignore[assignment]

    result = tools.dispatch(
        "run_aggregation",
        {"database_name": "BigQuery-GA", "schema_name": "business_data", "table_name": "orders", "aggregation": "count", "relative_days": 7, "time_grain": "day"},
    )

    assert result["status"] == "completed"
    assert dataset_payloads[0]["database"] == 19
    assert dataset_payloads[0]["query"]["source-table"] == 1332


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


def test_loop_injects_memory_and_skills_context() -> None:
    transport = _ScriptedTransport(["完成。"])

    run_tool_loop(
        "有哪些数据库？",
        [],
        transport,
        _tools(),
        memory_context="长期记忆上下文：用户偏好中文。",
        skills_context="可用任务技能：优先查询元数据。",
    )

    system_prompt = transport.calls[0]["messages"][0]["content"]
    assert "长期记忆上下文" in system_prompt
    assert "可用任务技能" in system_prompt


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


def test_agent_tools_reuse_cached_metabase_client() -> None:
    from metabase_agent.agent.tools import _client_cache

    _client_cache.clear()
    settings = Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="k", METABASE_BASE_URL="https://mb.test")
    a = AgentTools(settings, dry_run=False)
    b = AgentTools(settings, dry_run=False)

    assert a.client() is b.client()


class _CountingTools:
    def __init__(self) -> None:
        self.calls = 0

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {"status": "completed", "databases": ["a"]}


def test_loop_detects_repeated_identical_tool_call() -> None:
    transport = _ScriptedTransport(
        [
            [ToolCall(id="c1", name="list_databases", arguments={})],
            [ToolCall(id="c2", name="list_databases", arguments={})],
            "完成。",
        ]
    )
    tools = _CountingTools()

    outcome = run_tool_loop("q", [], transport, tools)

    assert tools.calls == 1
    assert any(event.get("step") == "tool.repeated" for event in outcome.trace)
    assert outcome.answer == "完成。"


def test_iter_tool_loop_streams_events_then_outcome() -> None:
    from metabase_agent.agent.tool_loop import LoopOutcome, iter_tool_loop

    transport = _ScriptedTransport(
        [
            [ToolCall(id="c1", name="list_databases", arguments={})],
            "有 3 个库。",
        ]
    )

    events = list(iter_tool_loop("有哪些库？", [], transport, _tools()))

    kinds = [kind for kind, _ in events]
    assert "trace" in kinds
    assert kinds[-1] == "outcome"
    assert isinstance(events[-1][1], LoopOutcome)
    assert events[-1][1].answer == "有 3 个库。"


def test_loop_suspends_with_database_name_and_defers_parallel_calls() -> None:
    transport = _ScriptedTransport(
        [
            [
                ToolCall(id="c1", name="run_sql", arguments={"sql": "SELECT 1", "database_name": "warehouse"}),
                ToolCall(id="c2", name="list_databases", arguments={}),
            ],
        ]
    )

    outcome = run_tool_loop("跑一下", [], transport, _tools())

    assert outcome.status == "requires_approval"
    assert outcome.pending_database_name == "warehouse"
    # Every tool_call in the suspended batch (except run_sql itself) must have
    # a response message, or the resumed conversation is rejected by the API.
    tool_messages = [message for message in outcome.messages if message.get("role") == "tool"]
    assert any(message["tool_call_id"] == "c2" for message in tool_messages)


def test_loop_resume_executes_approved_sql_on_approved_database() -> None:
    captured: dict[str, Any] = {}

    class _StubClient:
        def list_databases(self) -> Any:
            return {"data": [{"id": 7, "name": "warehouse"}, {"id": 19, "name": "BigQuery-GA"}]}

        def execute_native_query(self, database_id: int, sql: str) -> dict[str, Any]:
            captured["database_id"] = database_id
            return {"status": "completed", "row_count": 0, "data": {"cols": [], "rows": []}}

    tools = AgentTools(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="k", METABASE_BIGQUERY_DATABASE_ID=19), dry_run=False)
    tools._client = _StubClient()  # type: ignore[assignment]
    transport = _ScriptedTransport(["已执行。"])

    outcome = run_tool_loop(
        "",
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "run_sql", "arguments": {"sql": "SELECT 1", "database_name": "warehouse"}}]}],
        transport,
        tools,
        approved_sql="SELECT 1",
        approved_database_name="warehouse",
        approved_tool_call_id="c1",
    )

    assert outcome.status == "completed"
    # The approved SQL must run on the database the user reviewed, not the default.
    assert captured["database_id"] == 7


def test_run_sql_rejects_unknown_database_instead_of_falling_back() -> None:
    class _StubClient:
        def list_databases(self) -> Any:
            return {"data": [{"id": 19, "name": "BigQuery-GA"}]}

        def execute_native_query(self, database_id: int, sql: str) -> dict[str, Any]:  # pragma: no cover
            raise AssertionError("must not execute against a fallback database")

    tools = AgentTools(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="k"), dry_run=False)
    tools._client = _StubClient()  # type: ignore[assignment]

    result = tools.dispatch("run_sql", {"sql": "SELECT 1", "database_name": "no-such-db"})

    assert result["status"] == "not_found"


def test_dispatch_turns_network_errors_into_tool_failures() -> None:
    class _StubClient:
        def list_databases(self) -> Any:
            raise httpx.ConnectError("boom")

    tools = AgentTools(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="k"), dry_run=False)
    tools._client = _StubClient()  # type: ignore[assignment]

    result = tools.dispatch("list_databases", {})

    assert result["status"] == "failed"
    assert "ConnectError" in result["error"]


def test_chat_httpx_adapter_parses_tool_calls_without_sdk_headers(monkeypatch) -> None:
    from metabase_agent.semantics import llm_client

    captured: dict[str, Any] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "list_databases", "arguments": "{}"}}
                            ],
                        }
                    }
                ]
            }

    def _fake_post(url: str, **kwargs: Any) -> _Resp:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(llm_client.httpx, "post", _fake_post)
    transport = llm_client.ChatHttpxToolTransport(Settings(OPENAI_API_KEY="k", OPENAI_WIRE_API="chat_completions_httpx"))

    reply = transport.complete([{"role": "user", "content": "dbs?"}], tool_schemas())

    assert isinstance(reply, list)
    assert reply[0].name == "list_databases"
    assert captured["url"].endswith("/chat/completions")
    # No SDK fingerprint headers — only auth and content type, so gateway
    # WAFs that 403 the OpenAI SDK accept these requests.
    assert set(captured["headers"]) == {"Authorization", "Content-Type"}
    assert captured["json"]["tools"][0]["type"] == "function"


def test_chat_httpx_adapter_parses_final_text(monkeypatch) -> None:
    from metabase_agent.semantics import llm_client

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "有 3 个库。"}}]}

    monkeypatch.setattr(llm_client.httpx, "post", lambda url, **kwargs: _Resp())
    transport = llm_client.ChatHttpxToolTransport(Settings(OPENAI_API_KEY="k", OPENAI_WIRE_API="chat_completions_httpx"))

    reply = transport.complete([{"role": "user", "content": "dbs?"}], tool_schemas())

    assert reply == "有 3 个库。"


def test_build_tool_transport_selects_chat_httpx(monkeypatch) -> None:
    from metabase_agent.semantics import llm_client

    transport = llm_client.build_tool_transport(Settings(OPENAI_API_KEY="k", OPENAI_WIRE_API="chat_completions_httpx"))

    assert isinstance(transport, llm_client.ChatHttpxToolTransport)
