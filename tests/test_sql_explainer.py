from __future__ import annotations

import pytest

from metabase_agent.config.settings import Settings
from metabase_agent.semantics.sql_explainer import explain_sql_with_llm, structural_sql_summary


HOURLY_ACTIVE_USERS_SQL = """SELECT
  TIMESTAMP_TRUNC(create_time, HOUR, 'US/Pacific') AS hour,
  COUNT(DISTINCT user_id) AS active_users
FROM `business_data.aigc_sessions`
WHERE status = 3
GROUP BY hour
ORDER BY hour"""


# The hardcoded explanation that used to leak for ANY input SQL.
_LEAKED_PHRASES = ("发票", "汇率", "JPY", "KRW", "moths", "payer_count", "退款")


def test_structural_summary_describes_the_actual_sql() -> None:
    summary = structural_sql_summary(HOURLY_ACTIVE_USERS_SQL)

    assert "business_data.aigc_sessions" in summary
    assert "COUNT" in summary
    assert "DISTINCT" in summary or "去重" in summary
    assert "GROUP BY" in summary or "分组" in summary
    assert "TIMESTAMP_TRUNC" in summary or "时间粒度" in summary


def test_structural_summary_never_leaks_hardcoded_invoice_explanation() -> None:
    summary = structural_sql_summary(HOURLY_ACTIVE_USERS_SQL)

    for phrase in _LEAKED_PHRASES:
        assert phrase not in summary


def test_structural_summary_handles_sql_without_table() -> None:
    summary = structural_sql_summary("SELECT 1 AS ok")

    # Must not invent tables, and must not leak the invoice text.
    assert "未能" in summary or "no table" in summary.lower()
    for phrase in _LEAKED_PHRASES:
        assert phrase not in summary


def test_explain_sql_with_llm_requires_api_key() -> None:
    with pytest.raises(Exception):
        explain_sql_with_llm("SELECT 1", Settings(OPENAI_API_KEY=""))


def test_explain_sql_with_llm_uses_chat_completions(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Message:
        content = "这条 SQL 统计每小时去重活跃用户。"

    class _Choice:
        message = _Message()

    class _Completion:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs: object) -> _Completion:
            captured.update(kwargs)
            return _Completion()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs

        chat = _Chat()

    monkeypatch.setattr("metabase_agent.semantics.sql_explainer.OpenAI", _FakeOpenAI)

    answer = explain_sql_with_llm(
        HOURLY_ACTIVE_USERS_SQL,
        Settings(OPENAI_API_KEY="test-key", OPENAI_WIRE_API="chat_completions"),
    )

    assert answer == "这条 SQL 统计每小时去重活跃用户。"
    messages = captured["messages"]
    assert isinstance(messages, list)
    # The actual SQL must be sent to the model (no hardcoded prompt).
    assert any("aigc_sessions" in str(message.get("content", "")) for message in messages)
