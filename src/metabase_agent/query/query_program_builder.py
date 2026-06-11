from __future__ import annotations

from datetime import date, timedelta
from typing import Any, cast

from metabase_agent.query.query_planner import QueryPlan


def build_table_aggregation_program(
    table_id: int,
    function: str,
    field_id: int | None = None,
    limit: int = 200,
    date_field_id: int | None = None,
    relative_days: int | None = None,
    time_grain: str | None = None,
) -> dict[str, Any]:
    aggregation: list[Any] = [function]
    if function != "count" and field_id is not None:
        aggregation.append(["field", field_id])
    operations: list[list[Any]] = []
    if date_field_id is not None and relative_days is not None:
        operations.append(["filter", ["time-interval", ["field", date_field_id], -relative_days, "day"]])
    operations.append(["aggregate", aggregation])
    if date_field_id is not None and time_grain:
        temporal_bucket = ["with-temporal-bucket", ["field", date_field_id], time_grain]
        operations.append(["breakout", temporal_bucket])
        operations.append(["order-by", temporal_bucket, "asc"])
    operations.append(["limit", limit])
    return {
        "source": {"type": "table", "id": table_id},
        "operations": operations,
    }


def build_program(plan: QueryPlan | dict[str, Any], inspected_entity: dict[str, Any] | None = None) -> dict[str, Any]:
    operations: list[list[Any]] = [["limit", plan.get("limit", 200)]]
    default_time_field = None
    if inspected_entity:
        default_time_field = inspected_entity.get("default_time_dimension_field_id")

    if plan.get("intent") in {"metric_trend", "comparison"} and default_time_field:
        operations.insert(0, ["breakout", ["with-temporal-bucket", ["field", default_time_field], plan.get("time_grain") or "day"]])

    return {
        "source": {"type": plan["source_type"], "id": plan["source_id"]},
        "operations": operations,
    }


def _table_aggregation_v1_payload(program: dict[str, Any]) -> dict[str, Any]:
    aggregate = next(operation[1] for operation in program["operations"] if operation and operation[0] == "aggregate")
    limit = next((operation[1] for operation in program["operations"] if operation and operation[0] == "limit"), 200)
    aggregation: dict[str, Any] = {"function": aggregate[0]}
    if len(aggregate) > 1:
        aggregation["field_id"] = aggregate[1][1]
    payload: dict[str, Any] = {"table_id": program["source"]["id"], "aggregations": [aggregation], "limit": limit}
    filters: list[dict[str, Any]] = []
    group_by: list[dict[str, Any]] = []
    agent_field_ids = cast(dict[int, str], program.get("agent_field_ids", {}))
    for operation in program["operations"]:
        if not operation:
            continue
        if operation[0] == "filter":
            expression = operation[1]
            if expression[0] == "time-interval":
                field_id = int(expression[1][1])
                if expression[2] >= 0 or expression[3] != "day":
                    raise ValueError("v1 fallback only supports negative day time intervals")
                cutoff = (date.today() - timedelta(days=abs(int(expression[2])))).isoformat()
                tomorrow = (date.today() + timedelta(days=1)).isoformat()
                agent_field_id = agent_field_ids.get(field_id, str(field_id))
                filters.append({"field_id": agent_field_id, "operation": "greater-than-or-equal", "value": cutoff})
                filters.append({"field_id": agent_field_id, "operation": "less-than", "value": tomorrow})
        if operation[0] == "breakout":
            expression = operation[1]
            if expression[0] == "with-temporal-bucket":
                field_id = int(expression[1][1])
                group_by.append({"field_id": agent_field_ids.get(field_id, str(field_id)), "field_granularity": expression[2]})
    if filters:
        payload["filters"] = filters
    if group_by:
        payload["group_by"] = group_by
    return payload
