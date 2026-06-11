from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from metabase_agent.cli.app import app
from metabase_agent.config.settings import get_settings

runner = CliRunner()


def test_cli_ask_dry_run_uses_pipeline() -> None:
    get_settings.cache_clear()
    result = runner.invoke(app, ["ask", "当前有几个数据库？", "--dry-run"])

    assert result.exit_code == 0
    assert "数据库" in result.stdout


def test_cli_ask_uses_tool_loop_when_configured(monkeypatch) -> None:
    from metabase_agent.agent.tool_loop import ToolCall

    monkeypatch.setenv("AGENT_MODE", "tools")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("AGENT_DRY_RUN", "false")
    get_settings.cache_clear()

    class _Transport:
        def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
            if any(m.get("role") == "tool" for m in messages):
                return "工具循环答复。"
            return [ToolCall(id="c1", name="list_databases", arguments={})]

    monkeypatch.setattr("metabase_agent.cli.app.build_tool_transport", lambda settings: _Transport())
    monkeypatch.setattr("metabase_agent.cli.app.AgentTools", lambda settings, dry_run: __import__("metabase_agent.agent.tools", fromlist=["AgentTools"]).AgentTools(settings, dry_run=True))

    result = runner.invoke(app, ["ask", "有哪些数据库"])

    assert result.exit_code == 0
    assert "工具循环答复" in result.stdout
    get_settings.cache_clear()


def test_cli_version_prints_package_version() -> None:
    get_settings.cache_clear()
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_ping_reports_ok_when_checks_pass(monkeypatch) -> None:
    monkeypatch.setenv("METABASE_BASE_URL", "https://mb.test")
    monkeypatch.setenv("METABASE_API_KEY", "mbk")
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    get_settings.cache_clear()

    monkeypatch.setattr("metabase_agent.cli.app.MetabaseClient.ping", lambda self: {"status": "ok"})
    monkeypatch.setattr("metabase_agent.cli.app.complete", lambda system, user, settings: "OK")

    result = runner.invoke(app, ["ping"])

    assert result.exit_code == 0
    assert "[OK] metabase" in result.stdout
    assert "[OK] openai" in result.stdout
    get_settings.cache_clear()


def test_cli_ping_fails_when_metabase_unreachable(monkeypatch) -> None:
    monkeypatch.setenv("METABASE_BASE_URL", "https://mb.test")
    monkeypatch.setenv("METABASE_API_KEY", "mbk")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    def _boom(self) -> dict:
        raise RuntimeError("connection refused")

    monkeypatch.setattr("metabase_agent.cli.app.MetabaseClient.ping", _boom)

    result = runner.invoke(app, ["ping"])

    assert result.exit_code == 1
    assert "[FAIL] metabase" in result.stdout
    get_settings.cache_clear()
