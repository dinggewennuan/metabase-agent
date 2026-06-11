from __future__ import annotations

from typing import Any, Literal, TypedDict


class QueryPlan(TypedDict):
    source_type: Literal["metric", "table"]
    source_id: int
    source_name: str
    intent: str
    time_grain: str | None
    limit: int
    requires_clarification: bool
    clarification_question: str | None


def build_query_plan(parsed_intent: dict[str, Any], entity: dict[str, Any] | None) -> QueryPlan:
    if entity is None:
        return {
            "source_type": "metric",
            "source_id": 0,
            "source_name": "",
            "intent": parsed_intent["intent"],
            "time_grain": parsed_intent.get("time_grain"),
            "limit": 200,
            "requires_clarification": True,
            "clarification_question": "没有找到明确的数据指标，请指定要查询的 Metric 或 Table。",
        }

    return {
        "source_type": entity.get("type", "metric"),
        "source_id": int(entity["id"]),
        "source_name": str(entity.get("name") or entity.get("display_name") or entity["id"]),
        "intent": parsed_intent["intent"],
        "time_grain": parsed_intent.get("time_grain"),
        "limit": 200,
        "requires_clarification": False,
        "clarification_question": None,
    }
