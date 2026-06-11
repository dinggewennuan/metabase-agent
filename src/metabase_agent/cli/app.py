from __future__ import annotations

import json
import logging

import typer
import uvicorn

from metabase_agent.agent.graph import build_graph
from metabase_agent.config.settings import get_settings

app = typer.Typer(help="Metabase semantic analytics agent")


@app.callback()
def main() -> None:
    """Metabase semantic analytics agent."""
    logging.basicConfig(level=logging.INFO, format="[metabase-agent] %(message)s")


@app.command()
def ask(question: str, dry_run: bool = typer.Option(False, help="Use deterministic local sample data.")) -> None:
    settings = get_settings()
    use_dry_run = dry_run or settings.agent_dry_run
    graph = build_graph(settings)
    result = graph.invoke({"question": question, "dry_run": use_dry_run})
    typer.echo(result["answer"])
    typer.echo(json.dumps({"query_plan": result.get("query_plan"), "program": result.get("program")}, ensure_ascii=False, indent=2))


@app.command()
def web(host: str = "127.0.0.1", port: int = 8765) -> None:
    uvicorn.run("metabase_agent.api.app:app", host=host, port=port, reload=False)
