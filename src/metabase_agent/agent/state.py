from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    dry_run: bool
    sql_approved: bool
    parsed_intent: dict[str, Any]
    search_result: dict[str, Any]
    selected_entity: dict[str, Any] | None
    inspected_entity: dict[str, Any] | None
    query_plan: dict[str, Any]
    table_aggregation: dict[str, Any]
    program: dict[str, Any]
    policy_result: dict[str, Any]
    query_result: dict[str, Any]
    answer: str
    trace: list[dict[str, Any]]
