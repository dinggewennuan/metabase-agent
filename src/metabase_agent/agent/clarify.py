"""Builders for follow-up suggestion prompts when a query needs clarification."""
from __future__ import annotations

from typing import Any


def _table_clarification_suggestions(intent: str, table_names: list[str], schema_name: str | None, aggregation_function: str | None, relative_days: int | None, time_grain: str | None) -> list[str]:
    suggestions: list[str] = []
    for table_name in table_names[:8]:
        table_ref = f"{schema_name} 下{table_name}" if schema_name else table_name
        if intent == "table_field_list":
            suggestions.append(f"{table_ref} 这个表有哪些字段？")
            continue
        aggregate = aggregation_function or "count"
        if relative_days and time_grain == "day":
            suggestions.append(f"{table_ref} 最近{relative_days}天的每天的数据{aggregate}")
        else:
            suggestions.append(f"{table_ref} 这个表的数据{aggregate}")
    return suggestions


def _table_candidate_suggestions(intent: str, candidates: list[tuple[dict[str, Any], int]], schema_name: str | None, aggregation_function: str | None, relative_days: int | None, time_grain: str | None) -> list[str]:
    return _table_clarification_suggestions(
        intent,
        [str(table.get("name")) for table, _score in candidates[:5] if table.get("name")],
        schema_name,
        aggregation_function,
        relative_days,
        time_grain,
    )


def _database_clarification_suggestions(database_names: list[str], table_name: str, aggregation_function: str | None, relative_days: int | None, time_grain: str | None) -> list[str]:
    suggestions: list[str] = []
    for database_name in database_names[:8]:
        if aggregation_function:
            if relative_days and time_grain == "day":
                suggestions.append(f"查询{database_name} 下{table_name} 最近{relative_days}天的每天的数据{aggregation_function}")
            else:
                suggestions.append(f"查询{database_name} 下{table_name} 这个表的数据{aggregation_function}")
        else:
            suggestions.append(f"查询{database_name} 下{table_name} 这个表有哪些字段？")
    return suggestions
