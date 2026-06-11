from __future__ import annotations

from typing import Any

from metabase_agent.agent.state import AgentState


def append_trace(state: AgentState, event: dict[str, Any]) -> list[dict[str, Any]]:
    trace = [*state.get("trace", []), event]
    print(f"[metabase-agent] {event}")
    return trace
