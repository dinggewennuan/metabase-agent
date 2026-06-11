from __future__ import annotations

import logging
from typing import Any

from metabase_agent.agent.state import AgentState

logger = logging.getLogger("metabase_agent")


def append_trace(state: AgentState, event: dict[str, Any]) -> list[dict[str, Any]]:
    trace = [*state.get("trace", []), event]
    logger.info("%s", event)
    return trace
