from __future__ import annotations

import json
import logging

import typer
import uvicorn

from metabase_agent.agent.graph import build_graph
from metabase_agent.agent.tool_loop import run_tool_loop
from metabase_agent.agent.tools import AgentTools
from metabase_agent.config.settings import get_settings
from metabase_agent.semantics.llm_client import build_tool_transport

app = typer.Typer(help="Metabase semantic analytics agent")


@app.callback()
def main() -> None:
    """Metabase semantic analytics agent."""
    logging.basicConfig(level=logging.INFO, format="[metabase-agent] %(message)s")


@app.command()
def ask(question: str, dry_run: bool = typer.Option(False, help="Use deterministic local sample data.")) -> None:
    settings = get_settings()
    use_dry_run = dry_run or settings.agent_dry_run
    if settings.agent_mode == "tools" and settings.openai_api_key and not use_dry_run:
        tools = AgentTools(settings, dry_run=use_dry_run)
        outcome = run_tool_loop(question, [], build_tool_transport(settings), tools)
        typer.echo(outcome.answer)
        if outcome.status == "requires_approval":
            typer.echo(f"[需要授权] 待执行 SQL：\n{outcome.pending_sql}\n（CLI 为单次执行，不支持交互式授权；请在 Web 端确认，或改用 dry-run 预览。）")
        return
    graph = build_graph(settings)
    result = graph.invoke({"question": question, "dry_run": use_dry_run})
    typer.echo(result["answer"])
    typer.echo(json.dumps({"query_plan": result.get("query_plan"), "program": result.get("program")}, ensure_ascii=False, indent=2))


@app.command()
def web(host: str = "127.0.0.1", port: int = 8765) -> None:
    settings = get_settings()
    if not settings.agent_dry_run and not settings.agent_api_token:
        logging.getLogger("metabase_agent").warning(
            "running in real mode without AGENT_API_TOKEN: /api endpoints are unauthenticated. "
            "Set AGENT_API_TOKEN (and AGENT_REQUIRE_TOKEN=true) before exposing beyond localhost."
        )
    uvicorn.run("metabase_agent.api.app:app", host=host, port=port, reload=False)
