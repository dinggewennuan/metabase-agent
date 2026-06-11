from datetime import date, timedelta

from metabase_agent.agent.graph import _table_aggregation_v1_payload
from metabase_agent.policy.query_policy import check_program
from metabase_agent.query.query_planner import build_query_plan
from metabase_agent.query.query_program_builder import (
    build_program,
    build_table_aggregation_program,
)


def test_metric_trend_program_uses_default_time_dimension() -> None:
    plan = build_query_plan(
        {"intent": "metric_trend", "time_grain": "day"},
        {"type": "metric", "id": 10, "name": "Total Revenue"},
    )

    program = build_program(plan, {"default_time_dimension_field_id": 305})

    assert program["source"] == {"type": "metric", "id": 10}
    assert ["breakout", ["with-temporal-bucket", ["field", 305], "day"]] in program["operations"]
    assert check_program(program)["allowed"] is True


def test_policy_rejects_large_limit() -> None:
    result = check_program({"source": {"type": "metric", "id": 1}, "operations": [["limit", 500]]})

    assert result == {"allowed": False, "reason": "limit must be <= 200"}


def test_table_count_program_uses_aggregate_operation() -> None:
    program = build_table_aggregation_program(11, "count")

    assert program == {"source": {"type": "table", "id": 11}, "operations": [["aggregate", ["count"]], ["limit", 200]]}
    assert check_program(program)["allowed"] is True


def test_table_sum_program_uses_field_ref() -> None:
    program = build_table_aggregation_program(11, "sum", 3)

    assert program == {"source": {"type": "table", "id": 11}, "operations": [["aggregate", ["sum", ["field", 3]]], ["limit", 200]]}
    assert check_program(program)["allowed"] is True


def test_table_aggregation_v1_payload_translates_program() -> None:
    program = build_table_aggregation_program(11, "sum", 3)

    assert _table_aggregation_v1_payload(program) == {"table_id": 11, "aggregations": [{"function": "sum", "field_id": 3}], "limit": 200}


def test_table_recent_daily_count_program_uses_filter_and_breakout() -> None:
    program = build_table_aggregation_program(11, "count", date_field_id=2, relative_days=7, time_grain="day")

    assert program == {
        "source": {"type": "table", "id": 11},
        "operations": [
            ["filter", ["time-interval", ["field", 2], -7, "day"]],
            ["aggregate", ["count"]],
            ["breakout", ["with-temporal-bucket", ["field", 2], "day"]],
            ["order-by", ["with-temporal-bucket", ["field", 2], "day"], "asc"],
            ["limit", 200],
        ],
    }
    assert check_program(program)["allowed"] is True


def test_table_recent_daily_count_v1_payload_translates_program() -> None:
    program = build_table_aggregation_program(11, "count", date_field_id=2, relative_days=7, time_grain="day")
    program["agent_field_ids"] = {2: "t11-2"}

    assert _table_aggregation_v1_payload(program) == {
        "table_id": 11,
        "aggregations": [{"function": "count"}],
        "limit": 200,
        "filters": [
            {"field_id": "t11-2", "operation": "greater-than-or-equal", "value": (date.today() - timedelta(days=7)).isoformat()},
            {"field_id": "t11-2", "operation": "less-than", "value": (date.today() + timedelta(days=1)).isoformat()},
        ],
        "group_by": [{"field_id": "t11-2", "field_granularity": "day"}],
    }
