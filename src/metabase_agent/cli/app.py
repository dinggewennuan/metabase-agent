from __future__ import annotations

import json
import logging
from typing import Any

import typer
import uvicorn

from metabase_agent.agent.graph import build_graph
from metabase_agent.agent.tool_loop import run_tool_loop
from metabase_agent.agent.tools import AgentTools
from metabase_agent.config.settings import Settings, get_settings
from metabase_agent.memory import build_memory_manager
from metabase_agent.semantics.llm_client import build_tool_transport, complete
from metabase_agent.skills import build_skill_registry
from metabase_agent.tools.metabase.client import MetabaseClient

app = typer.Typer(help="Metabase semantic analytics agent")


@app.callback()
def main() -> None:
    """Metabase semantic analytics agent."""
    logging.basicConfig(level=logging.INFO, format="[metabase-agent] %(message)s")


@app.command()
def ask(question: str, dry_run: bool = typer.Option(False, help="Use deterministic local sample data.")) -> None:
    settings = get_settings()
    use_dry_run = dry_run or settings.agent_dry_run
    tenant_id, user_id, memory_context, skills_context = _cli_contexts(settings, question)
    if settings.agent_mode == "tools" and settings.openai_api_key and not use_dry_run:
        tools = AgentTools(settings, dry_run=use_dry_run)
        outcome = run_tool_loop(question, [], build_tool_transport(settings), tools, memory_context=memory_context, skills_context=skills_context)
        _record_cli_memory(settings, tenant_id, user_id, question, outcome.answer, {"query_plan": {"intent": "tool_loop", "status": outcome.status}, "query_result": outcome.last_result or {"status": outcome.status}})
        typer.echo(outcome.answer)
        if outcome.status == "requires_approval":
            typer.echo(f"[需要授权] 待执行 SQL：\n{outcome.pending_sql}\n（CLI 为单次执行，不支持交互式授权；请在 Web 端确认，或改用 dry-run 预览。）")
        return
    graph = build_graph(settings)
    result = graph.invoke({"question": question, "dry_run": use_dry_run, "tenant_id": tenant_id, "user_id": user_id, "memory_context": memory_context, "skills_context": skills_context})
    _record_cli_memory(settings, tenant_id, user_id, question, str(result.get("answer", "")), result)
    typer.echo(result["answer"])
    typer.echo(json.dumps({"query_plan": result.get("query_plan"), "program": result.get("program")}, ensure_ascii=False, indent=2))


@app.command()
def version() -> None:
    typer.echo(_package_version())


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        return _dist_version("metabase-agent-python")
    except PackageNotFoundError:
        from metabase_agent import __version__

        return __version__


@app.command()
def ping() -> None:
    checks = run_ping_checks(get_settings())
    for name, ok, detail in checks:
        typer.echo(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    if not all(ok for _name, ok, _detail in checks):
        raise typer.Exit(code=1)


def run_ping_checks(settings: Settings) -> list[tuple[str, bool, str]]:
    return [_ping_metabase(settings), _ping_openai(settings)]


def _cli_contexts(settings: Settings, question: str) -> tuple[str, str, str, str]:
    tenant_id = settings.agent_tenant_id or "default"
    user_id = settings.agent_user_id or "cli"
    memory_context = ""
    skills_context = ""
    try:
        memory_context = build_memory_manager(settings).load_context(tenant_id=tenant_id, user_id=user_id, query=question).rendered
    except Exception as exc:
        logging.getLogger("metabase_agent").warning("memory.context.load failed: %s", exc)
    try:
        skills_context = build_skill_registry(settings).render_context(question)
    except Exception as exc:
        logging.getLogger("metabase_agent").warning("skills.context.load failed: %s", exc)
    return tenant_id, user_id, memory_context, skills_context


def _record_cli_memory(settings: Settings, tenant_id: str, user_id: str, question: str, answer: str, result: dict[str, Any]) -> None:
    if not settings.agent_long_term_memory_enabled:
        return
    try:
        query_result = result.get("query_result")
        query_plan = result.get("query_plan")
        build_memory_manager(settings).record_interaction(
            tenant_id=tenant_id,
            user_id=user_id,
            question=question,
            answer=answer,
            query_result=query_result if isinstance(query_result, dict) else None,
            query_plan=query_plan if isinstance(query_plan, dict) else None,
        )
    except Exception as exc:
        logging.getLogger("metabase_agent").warning("memory.record_interaction failed: %s", exc)


def _ping_metabase(settings: Settings) -> tuple[str, bool, str]:
    if not settings.metabase_base_url or not settings.metabase_api_key:
        return ("metabase", False, "missing METABASE_BASE_URL / METABASE_API_KEY")
    try:
        MetabaseClient(settings.metabase_base_url, settings.metabase_api_key).ping()
        return ("metabase", True, settings.metabase_base_url)
    except Exception as exc:
        return ("metabase", False, f"{type(exc).__name__}: {exc}")


def _ping_openai(settings: Settings) -> tuple[str, bool, str]:
    if not settings.openai_api_key:
        return ("openai", False, "missing OPENAI_API_KEY")
    try:
        text = complete("只回 OK。", "ping", settings)
        return ("openai", True, f"{settings.openai_model} -> {text[:40]}")
    except Exception as exc:
        return ("openai", False, f"{type(exc).__name__}: {exc}")


@app.command()
def web(host: str = "127.0.0.1", port: int = 8765) -> None:
    settings = get_settings()
    if not settings.agent_dry_run and not settings.agent_api_token:
        logging.getLogger("metabase_agent").warning(
            "running in real mode without AGENT_API_TOKEN: /api endpoints are unauthenticated. "
            "Set AGENT_API_TOKEN (and AGENT_REQUIRE_TOKEN=true) before exposing beyond localhost."
        )
    uvicorn.run("metabase_agent.api.app:app", host=host, port=port, reload=False)
