from metabase_agent.semantics.llm_intent import _parse_responses_payload, _reasoning_effort


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
