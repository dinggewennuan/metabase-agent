from metabase_agent.semantics.llm_intent import (
    _parse_responses_payload,
    _reasoning_effort,
)


def test_parse_responses_payload_normalizes_intent() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"intent":"count_tables","database_name":"BigQuery-GA","schema_name":"business_data","table_name":null,"time_grain":null}',
                    }
                ],
            }
        ]
    }

    parsed = _parse_responses_payload(payload)

    assert parsed == {
        "intent": "database_table_count",
        "database_name": "BigQuery-GA",
        "schema_name": "business_data",
        "table_name": None,
        "time_grain": None,
    }


def test_parse_responses_payload_normalizes_sql_intent_alias() -> None:
    payload = {"output": [{"type": "message", "content": [{"type": "output_text", "text": '{"intent":"explain_sql"}'}]}]}

    parsed = _parse_responses_payload(payload)

    assert parsed == {"intent": "sql_explanation"}


def test_reasoning_effort_uses_highest_supported_value() -> None:
    assert _reasoning_effort("gpt-5") == "high"
    assert _reasoning_effort("gpt-5.5") == "xhigh"


def test_complete_uses_httpx_chat_wire_with_json_mode(monkeypatch) -> None:
    from metabase_agent.config.settings import Settings
    from metabase_agent.semantics import llm_client

    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": '{"intent": "database_table_count"}'}}]}

    def _fake_post(url: str, **kwargs: object) -> _Resp:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _Resp()

    monkeypatch.setattr(llm_client.httpx, "post", _fake_post)
    settings = Settings(OPENAI_API_KEY="k", OPENAI_WIRE_API="chat_completions_httpx")

    text = llm_client.complete("system", "user", settings, json_mode=True)

    assert text == '{"intent": "database_table_count"}'
    assert str(captured["url"]).endswith("/chat/completions")
    assert captured["json"]["response_format"] == {"type": "json_object"}  # type: ignore[index]
